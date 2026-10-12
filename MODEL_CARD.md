---
language:
- en
- tw
license: cc-by-nc-4.0
tags:
- translation
- tool-calling
- Akan
- Asante Twi
- gemma-4
- reasoning
- GLOSS-MT
base_model: google/gemma-4-E4B-it
datasets:
- ghanaopenai/twi-translation-tool-calling
metrics:
- chrf
- bleu
pipeline_tag: translation
---

# GLOSS-MT: Grounded Lexical Offline Search for Sequence Translation (Gemma 4 E4B)

**GLOSS-MT** (`ghanaopenai/gloss-mt`) is a 4-billion parameter state-of-the-art neural machine translation model for English to Asante Twi (Akan), fine-tuned on `google/gemma-4-E4B-it`. It pioneeringly combines **autonomous dictionary tool-calling** with **explicit linguistic reasoning** to solve lexical scarcity and hallucination in low-resource machine translation.

- **Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)
- **Supported by:** [Ghana NLP](https://ghananlp.org)
- **GitHub Repository:** [GhanaOpenAI/gloss-mt](https://github.com/GhanaOpenAI/gloss-mt)
- **Interactive Live Demo:** [Hugging Face Spaces: twi-translator](https://huggingface.co/spaces/ghanaopenai/twi-translator)
- **Base Model:** `google/gemma-4-E4B-it` (4B parameter compact multimodal/language foundation model)
- **Bundled Dictionary:** 545,633 English→Twi verified offline entries

---

## Benchmark Evaluation (nsanku-mt-benchmark)

Evaluated across the comprehensive document-level `nsanku-mt-benchmark` suite (**84 documents** spanning 4 heterogeneous domains) against leading commercial and open-source translation models:

### Overall Leaderboard (English → Asante Twi)

| Model | Track / Architecture | Parameters | **chrF2 (Avg)** | **BLEU** | Finance (3 Docs) | UDHR (Legal) | Watchtower (5 Docs) | Folktales (75 Docs) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Google Translate** | Commercial API | Cloud | **51.23** | 15.66 | 48.06 | 52.30 | 67.15 | 37.43 |
| **GLOSS-MT (Ours)** | **LLM + Offline Tool** | **4B** | **49.75** | **11.45** | **51.23** | **50.80** | **60.75** | **36.23** |
| **NLLB-200 3.3B** | Dedicated Open MT | 3.3B | **47.68** | 14.15 | 45.25 | 52.05 | 56.89 | 36.54 |
| **NLLB-200 1.3B** | Dedicated Open MT | 1.3B | **47.35** | 13.67 | 45.72 | 50.47 | 57.46 | 35.74 |
| **NLLB-200 600M** | Distilled Open MT | 600M | **46.71** | 12.75 | 44.10 | 50.17 | 57.06 | 35.51 |
| **Gemma 4 31B Base** | Foundation LLM | 31B | **45.44** | 11.95 | 45.97 | 41.97 | 57.57 | 36.26 |
| **MADLAD-400 10B** | Dedicated Open MT | 10B | **38.39** | 7.92 | 39.25 | 37.26 | 53.90 | 23.16 |
| **MADLAD-400 3B** | Dedicated Open MT | 3B | **36.97** | 6.70 | 37.43 | 36.60 | 50.66 | 23.19 |
| **Gemma 4 E4B Base** | Foundation LLM | 4B | **12.21** | 0.94 | 8.18 | 3.84 | 21.86 | 14.96 |

---

### Key Empirical Findings

1. **Outperforms All Open Translation Systems**: At just 4 billion parameters, **GLOSS-MT beats all open-source models** on the benchmark, including Meta's NLLB-200 3.3B (`49.75` vs `47.68`), Google's MADLAD-400 10B (`49.75` vs `38.39`), and Google's Gemma 4 31B Base (`49.75` vs `45.44`).
2. **Domain Champion on Technical Finance**: On national budgetary and fiscal documents (Ministry of Finance, 2021–2023), GLOSS-MT achieves **`51.23 chrF2`**, outperforming Google Translate (`48.06`), NLLB-200 3.3B (`45.25`), and Gemma 4 31B (`45.97`).
3. **Statutory & Legal Rights (UDHR)**: Achieves **`50.80 chrF2`**, completely eliminating the severe hallucination and verbose repetition exhibited by base LLMs (`41.97` on Gemma 31B).
4. **Religious Literature (Watchtower)**: Reaches **`60.75 chrF2`** and **`18.92 BLEU`**, outperforming all NLLB and MADLAD open models by +3 to +10 chrF points.
5. **Authentic Oral Folklore (Akan Folktales)**: Scores **`36.23 chrF2`** across 75 historical narrative stories (Rattray 1930) with exact sentence-bounded length fidelity (length ratio 1.19).

---

## How It Works: Tool-Calling + Reasoning Trajectory

GLOSS-MT executes translation through an explicit multi-step reasoning trajectory:

```
Input English Text
       │
       ▼
[1] Tool Call: <|tool_call>call:lookup_twi_translation{phrases: ["court", "decision"]}
       │
       ▼
[2] Offline Dictionary Lookup: 545,633 bilingual entries (no network required)
       │
       ▼
[3] Linguistic Reasoning: <channel|>thought Analysis of syntax, concord, and idioms...
       │
       ▼
[4] Final Translation: Accurate Asante Twi synthesis
```

---

## Quick Start & Usage

### 1. In-Process Python (with `transformers`)

Install the official package:
```bash
pip install git+https://github.com/GhanaOpenAI/gloss-mt.git
```

Run in Python:
```python
from twi_translate import TwiTranslator, TransformersBackend

# Load model locally
translator = TwiTranslator(TransformersBackend("ghanaopenai/gloss-mt"))

result = translator.translate("The court announced the decision yesterday.")
print("Translation:", result["translation"])
print("Reasoning:  ", result["reasoning"])
print("Lookups:    ", result["lookups"])
```

### 2. High-Throughput Serving (with `vLLM`)

```bash
# Start vLLM inference server
vllm serve ghanaopenai/gloss-mt \
    --served-model-name twi-translator \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --max-model-len 8192 \
    --port 8300
```

Translate via CLI or client:
```bash
twi-translate --base-url http://localhost:8300/v1 --model twi-translator -v "The queen-mother advised the youth to avoid illegal mining."
```

---

## Citation & License

- **License:** CC-BY-NC-4.0 (Non-commercial research and community deployment)
- **Citation:** Ghana Open AI & Ghana NLP (2026). *GLOSS-MT: Grounding Generative LLMs with External Dictionaries for Low-Resource Machine Translation*.
