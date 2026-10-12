# Machine Translation Benchmark Comparison: English → Asante Twi (Akan)

Comprehensive document-level benchmark evaluation on the `nsanku-mt-benchmark` suite across four distinct domains.

## Overall Leaderboard (English → Asante Twi)

| Model | Track | chrF2 (Avg) | BLEU | Finance (Budget) | UDHR | Watchtower (JW) | Folk-Tales | Docs Scored |
|---|---|---|---|---|---|---|---|---|
| **Gemini 3.8 Flash** | llm | **54.79** | 17.42 | 52.5 | 56.44 | 69.28 | 40.95 | 84 |
| **Google Translate (free endpoint)** | api | **51.23** | 15.66 | 48.06 | 52.3 | 67.15 | 37.43 | 84 |
| **Khaya AI** | api | **49.91** | 12.79 | 47.74 | 49.92 | 65.78 | 36.22 | 84 |
| **NLLB-200 3.3B** | open | **47.68** | 14.15 | 45.25 | 52.05 | 56.89 | 36.54 | 84 |
| **NLLB-200 1.3B** | open | **47.35** | 13.67 | 45.72 | 50.47 | 57.46 | 35.74 | 84 |
| **NLLB-200 600M (distilled)** | open | **46.71** | 12.75 | 44.1 | 50.17 | 57.06 | 35.51 | 84 |
| **Gemma 4 31B Instruct (Base)** | llm | **45.44** | 11.95 | 45.97 | 41.97 | 57.57 | 36.26 | 84 |
| **MADLAD-400 10B** | open | **38.39** | 7.92 | 39.25 | 37.26 | 53.9 | 23.16 | 84 |
| **MADLAD-400 3B** | open | **36.97** | 6.7 | 37.43 | 36.6 | 50.66 | 23.19 | 84 |
| *Gemma 4 E4B Instruct (Base)* | llm | *queued* | - | - | - | - | - | 84 |
| *GLOSS-MT (Gemma 4 E4B + Tool Calling)* | llm | *queued* | - | - | - | - | - | 84 |
| *GLOSS-MT (Gemma 4 E4B + Expanded Dict)* | llm | *queued* | - | - | - | - | - | 84 |

## Domain & Dataset Breakdown

### 1. Finance / Citizens' Budget (Ministry of Finance, 2021–2023)
Technical and administrative domain covering national budgetary allocations, policy frameworks, and fiscal indicators.

| Model | 2021 chrF2 | 2022 chrF2 | 2023 chrF2 | Finance Avg chrF2 |
|---|---|---|---|---|
| Gemini 3.8 Flash | 53.28 | 50.3 | 53.93 | **52.5** |
| Google Translate (free endpoint) | 49.12 | 46.47 | 48.6 | **48.06** |
| Khaya AI | 48.33 | 46.57 | 48.33 | **47.74** |
| Gemma 4 31B Instruct (Base) | 46.27 | 44.3 | 47.34 | **45.97** |
| NLLB-200 1.3B | 46.24 | 43.75 | 47.16 | **45.72** |
| NLLB-200 3.3B | 46.06 | 43.54 | 46.14 | **45.25** |
| NLLB-200 600M (distilled) | 45.36 | 41.72 | 45.21 | **44.1** |
| MADLAD-400 10B | 39.16 | 38.5 | 40.1 | **39.25** |
| MADLAD-400 3B | 37.93 | 36.31 | 38.04 | **37.43** |
| *Gemma 4 E4B Instruct (Base)* | *queued* | *queued* | *queued* | - |
| *GLOSS-MT (Gemma 4 E4B + Tool Calling)* | *training* | *training* | *training* | - |

### 2. Universal Declaration of Human Rights (UDHR)
Formal legalistic text translated into Ghanaian languages.

| Model | chrF2 | BLEU | Length Ratio | Words |
|---|---|---|---|---|
| Gemini 3.8 Flash | **56.44** | 10.15 | 0.949 | 2040 / 2098 |
| Google Translate (free endpoint) | **52.3** | 9.29 | 0.881 | 1909 / 2098 |
| NLLB-200 3.3B | **52.05** | 10.45 | 0.92 | 1989 / 2098 |
| NLLB-200 1.3B | **50.47** | 8.12 | 0.956 | 2073 / 2098 |
| NLLB-200 600M (distilled) | **50.17** | 7.88 | 0.887 | 1913 / 2098 |
| Khaya AI | **49.92** | 8.06 | 1.032 | 2265 / 2098 |
| Gemma 4 31B Instruct (Base) | **41.97** | 4.26 | 1.566 | 3480 / 2098 |
| MADLAD-400 10B | **37.26** | 2.82 | 1.658 | 3440 / 2098 |
| MADLAD-400 3B | **36.6** | 2.03 | 2.103 | 4558 / 2098 |
| *Gemma 4 E4B Instruct (Base)* | *queued* | *queued* | - | - |
| *GLOSS-MT (Gemma 4 E4B + Tool Calling)* | *training* | *training* | - | - |

### 3. Watchtower Study Editions (JW)
Moral, community, and religious prose.

| Model | JW Avg chrF2 | Documents Scored |
|---|---|---|
| Gemini 3.8 Flash | **69.28** | 5 |
| Google Translate (free endpoint) | **67.15** | 5 |
| Khaya AI | **65.78** | 5 |
| Gemma 4 31B Instruct (Base) | **57.57** | 5 |
| NLLB-200 1.3B | **57.46** | 5 |
| NLLB-200 600M (distilled) | **57.06** | 5 |
| NLLB-200 3.3B | **56.89** | 5 |
| MADLAD-400 10B | **53.9** | 5 |
| MADLAD-400 3B | **50.66** | 5 |
| *Gemma 4 E4B Instruct (Base)* | *queued* | 5 |
| *GLOSS-MT (Gemma 4 E4B + Tool Calling)* | *training* | 5 |

### 4. Akan-Ashanti Folk-Tales (Rattray, 1930)
Oral literature and narrative storytelling covering 75 traditional folktales in authentic Asante Twi.

| Model | Folktales Avg chrF2 | Documents Scored |
|---|---|---|
| Gemini 3.8 Flash | **40.95** | 75 |
| Google Translate (free endpoint) | **37.43** | 75 |
| Gemma 4 31B Instruct (Base) | **36.26** | 75 |
| Khaya AI | **36.22** | 75 |
| NLLB-200 3.3B | **36.54** | 75 |
| NLLB-200 1.3B | **35.74** | 75 |
| NLLB-200 600M (distilled) | **35.51** | 75 |
| MADLAD-400 10B | **23.16** | 75 |
| MADLAD-400 3B | **23.19** | 75 |
| *Gemma 4 E4B Instruct (Base)* | *queued* | 75 |
| *GLOSS-MT (Gemma 4 E4B + Tool Calling)* | *training* | 75 |

## Research Architecture & Next Benchmark Phase

1. **Gemma 4 E4B as Primary Research Flagship Model:**
   - With 4B parameters, `gemma-4-E4B-it` is highly accessible for resource-constrained environments, edge deployment, and low-latency translation pipelines.
   - SFT fine-tuning with LoRA ($r=32$) grounds the generative model with external dictionary lookups via tool calling (GLOSS-MT).

2. **Benchmark Chunking Correction:**
   - Previous tool runs on `folktales` suffered severe truncation due to a 2,500-character fallback split that exhausted generation token limits on unformatted narrative texts.
   - Benchmark splitting is now set to 400 characters (~60 words / 1–2 sentences) with a 4096 token generation window, eliminating truncation across all 75 folktale documents.
   - The full 84-document evaluation of `google/gemma-4-E4B-it` (Base) and `ghanaopenai/twi-translator-gemma4-e4b` (GLOSS-MT) will be executed upon completion of fine-tuning.
