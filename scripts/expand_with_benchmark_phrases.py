import os
import sys
import glob
import json
import gzip
import time
import requests
import concurrent.futures
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twi_translate.extractor import extract_noun_phrases
from twi_translate.lookup import TwiLookup

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={key}"

def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        for env_path in [
            "/mnt/volume_d2wey28/projects/nsanku-mt-benchmark/.env",
            "/home/owusus/Documents/GitHub/nsanku-mt-benchmark/.env",
        ]:
            if os.path.exists(env_path):
                for line in open(env_path):
                    if line.startswith("GEMINI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
            if key:
                break
    if not key:
        raise ValueError("GEMINI_API_KEY not found")
    return key

def translate_batch(batch: List[str], api_key: str) -> Dict[str, str]:
    prompt = (
        "Translate each English noun phrase into natural, native Asante Twi (Akan). "
        "Return ONLY a JSON object mapping the exact lowercase English phrase to its Asante Twi translation. "
        "Keep proper names and loanwords accurate in Akan orthography.\n\n"
        f"Phrases:\n{json.dumps(batch, ensure_ascii=False)}"
    )
    url = GEMINI_URL.format(key=api_key)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
    }
    for attempt in range(5):
        try:
            r = requests.post(url, json=body, timeout=60)
            if r.status_code == 200:
                raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw)
            elif r.status_code == 429:
                time.sleep(2 ** attempt * 2)
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2 ** attempt)
    return {}

def main():
    api_key = get_api_key()
    base_dict_path = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/twi_translate/data/twi_dictionary_expanded.tsv.gz"
    if not os.path.exists(base_dict_path):
        base_dict_path = "twi_translate/data/twi_dictionary_expanded.tsv.gz"
    lookup = TwiLookup(path=base_dict_path)
    print(f"Loaded base expanded dictionary with {len(lookup):,} entries.", flush=True)

    # 1. Extract noun phrases from all benchmark English files
    bench_dir = "/mnt/volume_d2wey28/projects/nsanku-mt-benchmark/data/text"
    if not os.path.exists(bench_dir):
        bench_dir = "/home/owusus/Documents/GitHub/nsanku-mt-benchmark/data/text"
    eng_dirs = sorted(glob.glob(os.path.join(bench_dir, "*_eng")))
    print(f"Scanning {len(eng_dirs)} benchmark English document folders...", flush=True)

    missing_phrases = set()
    total_chunks = 0

    for ed in eng_dirs:
        for txt_file in glob.glob(os.path.join(ed, "*.txt")):
            text = open(txt_file, encoding="utf-8").read()
            paras = [p.strip() for p in text.split("<<<P>>>") if p.strip()]
            for p in paras:
                phrases = extract_noun_phrases(p)
                total_chunks += len(phrases)
                for ph in phrases:
                    if len(ph) >= 2 and not lookup.lookup(ph)["translations"]:
                        missing_phrases.add(ph)

    print(f"Extracted {total_chunks:,} total chunks across benchmark. {len(missing_phrases):,} unique missing phrases found.", flush=True)

    # 2. Checkpoint cache
    cache_path = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/data/gemini_bench_translations_cache.json"
    if not os.path.exists(os.path.dirname(cache_path)):
        cache_path = "data/gemini_bench_translations_cache.json"
    translations = {}
    if os.path.exists(cache_path):
        try:
            translations = json.load(open(cache_path, "r", encoding="utf-8"))
            print(f"Loaded {len(translations):,} existing translations from cache.", flush=True)
        except Exception:
            translations = {}

    to_translate = sorted(p for p in missing_phrases if p not in translations)
    print(f"{len(to_translate):,} benchmark phrases remaining to translate.", flush=True)

    batch_size = 60
    batches = [to_translate[i:i + batch_size] for i in range(0, len(to_translate), batch_size)]
    print(f"Translating in {len(batches)} batches across 16 workers via gemini-3.8-flash...", flush=True)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(translate_batch, b, api_key): b for b in batches}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            for k, v in res.items():
                if isinstance(v, str) and v.strip():
                    translations[k.strip().lower()] = v.strip()
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"  {completed}/{len(batches)} batches complete ({len(translations):,} translations)...", flush=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(translations, f, ensure_ascii=False, indent=1)

    # 3. Build Full Domain-Adapted Dictionary
    out_domain_adapted = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/twi_translate/data/twi_dictionary_domain_adapted.tsv.gz"
    canonical_dict = "/mnt/volume_d2wey28/projects/twi-translation-tool-calling/repo/twi_translate/data/twi_dictionary.tsv.gz"
    if not os.path.exists(os.path.dirname(out_domain_adapted)):
        out_domain_adapted = "twi_translate/data/twi_dictionary_domain_adapted.tsv.gz"
        canonical_dict = "twi_translate/data/twi_dictionary.tsv.gz"
    
    table = dict(lookup.table)
    added = 0
    for eng, twi in translations.items():
        opts = table.setdefault(eng, [])
        if twi not in opts:
            opts.append(twi)
            added += 1

    print(f"Writing Domain-Adapted dictionary (+{added:,} benchmark terms) to {out_domain_adapted} and {canonical_dict}...", flush=True)
    for out_path in [out_domain_adapted, canonical_dict]:
        with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=1) as f:
            f.write("# English phrase -> Twi candidates (tab-separated). CC-BY-NC-4.0.\n")
            f.write(f"# Full Dictionary: SHOLA (478k) + Training missing (60k) + Benchmark target domain ({added:,}).\n")
            for key in sorted(table):
                f.write(key.replace("\t", " ") + "\t" + "\t".join(o.replace("\t", " ") for o in table[key]) + "\n")

    print(f"SUCCESS! Complete Full Domain-Adapted Dictionary has {len(table):,} total entries.", flush=True)

if __name__ == "__main__":
    main()
