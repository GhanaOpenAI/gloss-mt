"""Compile benchmark comparison results from nsanku-mt-benchmark into results/
for paper writing and evaluation analysis.
"""

import json
import os
import sys
from collections import defaultdict

import yaml

BENCHMARK_REPO = "/home/owusus/Documents/GitHub/nsanku-mt-benchmark"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    yaml_path = os.path.join(BENCHMARK_REPO, "benchmarks", "twi_asante.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} does not exist.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models_order = [
        "Google/gemini-3.8-flash",
        "Google/google-translate-free",
        "ghanaopenai/gloss-mt-nllb-3.3b",
        "ghanaopenai/gloss-mt-nllb-1.3b",
        "ghanaopenai/gloss-mt-nllb-600m",
        "KhayaAI/khaya-translate",
        "ghanaopenai/gloss-mt-translategemma-4b",
        "ghanaopenai/gloss-mt-opus-mt-en-tw",
        "facebook/nllb-200-3.3B",
        "facebook/nllb-200-1.3B",
        "facebook/nllb-200-distilled-600M",
        "Helsinki-NLP/opus-mt-en-tw",
        "google/madlad400-3b-mt",
    ]

    models_data = {}
    for b in data.get("benchmarks", []):
        m_id = b.get("model")
        fe = b.get("directions", {}).get("from_en", {})
        if fe:
            models_data[m_id] = {
                "name": b.get("name", m_id),
                "track": b.get("track", "open"),
                "chrf": fe.get("chrf", 0),
                "bleu": fe.get("bleu", 0),
                "length_ratio": fe.get("length_ratio", 0),
                "documents": fe.get("documents", 0),
                "per_dataset": fe.get("per_dataset", {}),
            }

    # Save JSON summary
    json_path = os.path.join(RESULTS_DIR, "benchmark_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(models_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON comparison to {json_path}")

    # Generate Markdown report
    md_lines = [
        "# Machine Translation Benchmark Comparison: English → Asante Twi (Akan)",
        "",
        "Comprehensive document-level benchmark evaluation on the `nsanku-mt-benchmark` suite across four distinct domains.",
        "",
        "## Overall Leaderboard (English → Asante Twi)",
        "",
        "| Model | Track | chrF2 (Avg) | BLEU | Finance (Budget) | UDHR | Watchtower (JW) | Folk-Tales | Docs Scored |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for m_id in models_order:
        if m_id not in models_data:
            continue
        m = models_data[m_id]
        p_ds = m["per_dataset"]
        fin = p_ds.get("finance", {}).get("chrf", "—")
        udhr = p_ds.get("udhr", {}).get("chrf", "—")
        jw = p_ds.get("jw", {}).get("chrf", "—")
        folk = p_ds.get("folktales", {}).get("chrf", "—")
        
        md_lines.append(
            f"| **{m['name']}** | {m['track']} | **{m['chrf']}** | {m['bleu']} | {fin} | {udhr} | {jw} | {folk} | {m['documents']} |"
        )

    # Detailed per-dataset breakdowns
    md_lines.extend([
        "",
        "## Domain & Dataset Breakdown",
        "",
        "### 1. Finance / Citizens' Budget (Ministry of Finance, 2021–2023)",
        "Technical and administrative domain covering national budgetary allocations, policy frameworks, and fiscal indicators.",
        "",
        "| Model | 2021 chrF2 | 2022 chrF2 | 2023 chrF2 | Finance Avg chrF2 |",
        "|---|---|---|---|---|",
    ])

    for m_id in models_order:
        if m_id not in models_data:
            continue
        m = models_data[m_id]
        fin_ds = m["per_dataset"].get("finance", {})
        if not fin_ds:
            continue
        p_doc = fin_ds.get("per_document", {})
        f21 = p_doc.get("finance_2021_twi_asante", {}).get("chrf", "—")
        f22 = p_doc.get("finance_2022_twi_asante", {}).get("chrf", "—")
        f23 = p_doc.get("finance_2023_twi_asante", {}).get("chrf", "—")
        f_avg = fin_ds.get("chrf", "—")
        md_lines.append(f"| {m['name']} | {f21} | {f22} | {f23} | **{f_avg}** |")

    md_lines.extend([
        "",
        "### 2. Universal Declaration of Human Rights (UDHR)",
        "Formal legalistic text translated into Ghanaian languages.",
        "",
        "| Model | chrF2 | BLEU | Length Ratio | Words |",
        "|---|---|---|---|---|",
    ])

    for m_id in models_order:
        if m_id not in models_data:
            continue
        m = models_data[m_id]
        udhr_ds = m["per_dataset"].get("udhr", {})
        if not udhr_ds:
            continue
        doc = udhr_ds.get("per_document", {}).get("udhr_twi_asante", {})
        md_lines.append(
            f"| {m['name']} | **{doc.get('chrf', '—')}** | {doc.get('bleu', '—')} | {doc.get('length_ratio', '—')} | {doc.get('hyp_words', '—')} / {doc.get('ref_words', '—')} |"
        )

    md_lines.extend([
        "",
        "### 3. Watchtower Study Editions (JW)",
        "Moral, community, and religious prose.",
        "",
        "| Model | JW Avg chrF2 | Documents Scored |",
        "|---|---|---|",
    ])

    for m_id in models_order:
        if m_id not in models_data:
            continue
        m = models_data[m_id]
        jw_ds = m["per_dataset"].get("jw", {})
        if not jw_ds:
            continue
        md_lines.append(f"| {m['name']} | **{jw_ds.get('chrf', '—')}** | {jw_ds.get('documents', 0)} |")

    md_lines.extend([
        "",
        "### 4. Akan-Ashanti Folk-Tales (Rattray, 1930)",
        "Oral literature and narrative storytelling covering 75 traditional folktales in authentic Asante Twi.",
        "",
        "| Model | Folktales Avg chrF2 | Documents Scored |",
        "|---|---|---|",
    ])

    for m_id in models_order:
        if m_id not in models_data:
            continue
        m = models_data[m_id]
        folk_ds = m["per_dataset"].get("folktales", {})
        if not folk_ds:
            continue
        md_lines.append(f"| {m['name']} | **{folk_ds.get('chrf', '—')}** | {folk_ds.get('documents', 0)} |")

    md_lines.extend([
        "",
        "## Key Research Findings for the Paper",
        "",
        "1. **Every GLOSS-MT fine-tune scores above its pretrained base on chrF2:** NLLB-200 3.3B 47.74 → 50.80 (+3.06), NLLB-200 1.3B 47.42 → 50.27 (+2.85), NLLB-200 600M 46.71 → 50.12 (+3.41), OPUS-MT en-tw 42.52 → 48.00 (+5.48). TranslateGemma 4B, which produces no Twi zero-shot, reaches 49.81.",
        "2. **Best open systems:** the three GLOSS-MT NLLB models score above Khaya AI (49.91), behind only Gemini 3.8 Flash and Google Translate.",
        "3. **Model size matters little once fine-tuned:** GLOSS-MT NLLB spans 50.12–50.80 across 600M–3.3B, and the 600M fine-tune scores above the base 3.3B model by +2.38.",
        "4. **The gain concentrates in contemporary prose** (Watchtower and Finance); Folk-Tales (1930 orthography) gain least.",
        "5. **BLEU falls slightly for the NLLB fine-tunes** (3.3B −1.04, 1.3B −0.86, 600M −0.51) while chrF2 rises; OPUS-MT gains BLEU (+1.20).",
    ])

    md_path = os.path.join(RESULTS_DIR, "benchmark_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown comparison report to {md_path}")


if __name__ == "__main__":
    main()
