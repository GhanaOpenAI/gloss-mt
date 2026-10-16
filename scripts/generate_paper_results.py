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
        "KhayaAI/khaya-translate",
        "ghanaopenai/gloss-mt-translategemma-4b",
        "ghanaopenai/gloss-mt",
        "ghanaopenai/gloss-mt-opus-mt-en-tw",
        "facebook/nllb-200-3.3B",
        "facebook/nllb-200-1.3B",
        "facebook/nllb-200-distilled-600M",
        "google/gemma-4-31B-it",
        "Helsinki-NLP/opus-mt-en-tw",
        "google/madlad400-3b-mt",
        "google/gemma-4-E4B-it",
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
        "1. **Tool-Grounded Augmentation vs. Base Direct Translation:**",
        "   - The unaugmented **Gemma 4 E4B Base** completely fails on Asante Twi translation across all domains, scoring an overall **12.21 chrF2** and suffering from catastrophic length inflation (ratio 14.13) and repetitive loops.",
        "   - **GLOSS-MT** elevates the 4B model to **49.75 chrF2** (+37.54 points overall), with a near-perfect length ratio of **1.19**.",
        "   - On Finance, GLOSS-MT gains **+43.05 chrF2 points** (`51.23` vs `8.18`).",
        "   - On UDHR (legal), GLOSS-MT gains **+46.96 chrF2 points** (`50.80` vs `3.84`), reducing word count from 49,341 degenerate words to 2,064 exact words.",
        "   - On Watchtower, GLOSS-MT gains **+38.89 chrF2 points** (`60.75` vs `21.86`).",
        "   - On Folktales, GLOSS-MT gains **+21.27 chrF2 points** (`36.23` vs `14.96`).",
        "",
        "2. **Surpassing 31B Base Model with 4B Parameters:**",
        "   - GLOSS-MT 4B outperforms **Gemma 4 31B Base** overall (**49.75 vs 45.44**, a +4.31 point advantage), demonstrating that external lexical grounding and reasoning far outweigh raw parameter scale for low-resource translation.",
        "   - On Finance, GLOSS-MT 4B (`51.23`) beats Gemma 4 31B (`45.97`) by **+5.26 points**.",
        "   - On UDHR, GLOSS-MT 4B (`50.80`) beats Gemma 4 31B (`41.97`) by **+8.83 points**.",
        "",
        "3. **Competitive Against Dedicated and Commercial Systems:**",
        "   - GLOSS-MT 4B outperforms every open dedicated MT model: NLLB-200 3.3B (+2.07), NLLB-200 1.3B (+2.40), NLLB-200 600M (+3.04), and MADLAD-400 10B (+11.36).",
        "   - On technical finance, GLOSS-MT 4B outperforms Google Translate (`51.23` vs `48.06`), placing second only to Gemini 3.8 Flash.",
        "",
        "4. **Tool-Free GLOSS-MT: Dictionary Terms Pretranslated In Place for a Dedicated MT Model:**",
        "   - **GLOSS-MT (NLLB-200 3.3B + Glossary)** looks up each chunk's noun phrases in the same dictionary, replaces the found phrases by their Twi translations inside the English, and translates the mixed text with NLLB-200 3.3B fine-tuned (LoRA) on such inputs. It scores **49.78 chrF2 / 13.93 BLEU** with a length ratio of **0.99**.",
        "   - **+2.04 chrF2** over the base NLLB-200 3.3B (`49.78` vs `47.74`), and level with the tool-calling GLOSS-MT Gemma 4 E4B (`49.75`) with **+2.48 BLEU** (`13.93` vs `11.45`), from one beam-search pass per chunk with no reasoning trace or tool loop.",
        "   - Largest gain on Watchtower: **`65.55`** vs `56.89` for base NLLB-200 3.3B (+8.66) and `60.75` for GLOSS-MT Gemma 4 E4B (+4.80).",
        "   - Finance `46.97` (+1.72 over base NLLB-200 3.3B, −4.26 below GLOSS-MT Gemma 4 E4B); UDHR `50.46` (−1.59 vs base NLLB-200 3.3B); Folktales `36.16` (−0.60).",
        "",
        "5. **The Same Pipeline on TranslateGemma 4B and OPUS-MT en-tw:**",
        "   - **GLOSS-MT (TranslateGemma 4B + Glossary)**, a LoRA fine-tune of TranslateGemma 4B on the same data, scores **49.81 chrF2 / 12.24 BLEU** (length ratio 1.06): −0.99 chrF2 below GLOSS-MT NLLB 3.3B on the benchmark although it led on validation (BLEU 33.73 vs 31.73).",
        "   - **GLOSS-MT (OPUS-MT en-tw + Glossary)**, a full fine-tune of the ~75M-parameter Marian model, scores **48.00 chrF2 / 10.91 BLEU**: **+5.48 chrF2** over the base OPUS-MT en-tw (`42.52`), with gains in every domain, and above base NLLB-200 3.3B on chrF2 (`47.74`).",
    ])

    md_path = os.path.join(RESULTS_DIR, "benchmark_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved Markdown comparison report to {md_path}")


if __name__ == "__main__":
    main()
