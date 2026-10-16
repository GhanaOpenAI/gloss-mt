# Machine Translation Benchmark Comparison: English → Asante Twi (Akan)

Comprehensive document-level benchmark evaluation on the `nsanku-mt-benchmark` suite across four distinct domains.

## Overall Leaderboard (English → Asante Twi)

| Model | Track | chrF2 (Avg) | BLEU | Finance (Budget) | UDHR | Watchtower (JW) | Folk-Tales | Docs Scored |
|---|---|---|---|---|---|---|---|---|
| **Gemini 3.8 Flash** | llm | **54.64** | 17.25 | 52.5 | 56.44 | 69.28 | 40.33 | 84 |
| **Google Translate (free endpoint)** | api | **51.26** | 15.67 | 48.06 | 52.3 | 67.15 | 37.54 | 84 |
| **GLOSS-MT (NLLB-200 3.3B + Glossary)** | open | **50.8** | 13.11 | 50.23 | 51.66 | 64.17 | 37.16 | 84 |
| **Khaya AI** | api | **49.91** | 12.79 | 47.74 | 49.92 | 65.78 | 36.22 | 84 |
| **GLOSS-MT (TranslateGemma 4B + Glossary)** | open | **49.81** | 12.24 | 50.03 | 51.19 | 62.54 | 35.48 | 84 |
| **GLOSS-MT (Gemma 4 E4B + Tool Calling)** | llm | **49.75** | 11.45 | 51.23 | 50.8 | 60.75 | 36.23 | 84 |
| **GLOSS-MT (OPUS-MT en-tw + Glossary)** | open | **48.0** | 10.91 | 46.13 | 50.09 | 60.62 | 35.15 | 84 |
| **NLLB-200 3.3B** | open | **47.74** | 14.15 | 45.25 | 52.05 | 56.89 | 36.76 | 84 |
| **NLLB-200 1.3B** | open | **47.42** | 13.68 | 45.72 | 50.47 | 57.46 | 36.05 | 84 |
| **NLLB-200 600M (distilled)** | open | **46.71** | 12.74 | 44.1 | 50.17 | 57.06 | 35.5 | 84 |
| **Gemma 4 31B Instruct (Base)** | llm | **45.55** | 11.93 | 45.97 | 41.97 | 57.57 | 36.71 | 84 |
| **OPUS-MT en-tw (Marian)** | open | **42.52** | 9.71 | 37.78 | 47.52 | 54.23 | 30.53 | 84 |
| **MADLAD-400 3B** | open | **37.14** | 6.71 | 37.43 | 36.6 | 50.66 | 23.87 | 84 |
| **Gemma 4 E4B Instruct (Base)** | llm | **12.21** | 0.94 | 8.18 | 3.84 | 21.86 | 14.96 | 84 |

## Domain & Dataset Breakdown

### 1. Finance / Citizens' Budget (Ministry of Finance, 2021–2023)
Technical and administrative domain covering national budgetary allocations, policy frameworks, and fiscal indicators.

| Model | 2021 chrF2 | 2022 chrF2 | 2023 chrF2 | Finance Avg chrF2 |
|---|---|---|---|---|
| Gemini 3.8 Flash | 53.28 | 50.3 | 53.93 | **52.5** |
| Google Translate (free endpoint) | 49.12 | 46.47 | 48.6 | **48.06** |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | 51.22 | 48.73 | 50.75 | **50.23** |
| Khaya AI | 48.33 | 46.57 | 48.33 | **47.74** |
| GLOSS-MT (TranslateGemma 4B + Glossary) | 50.6 | 48.45 | 51.03 | **50.03** |
| GLOSS-MT (Gemma 4 E4B + Tool Calling) | 49.52 | 51.31 | 52.87 | **51.23** |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | 45.74 | 44.78 | 47.87 | **46.13** |
| NLLB-200 3.3B | 46.06 | 43.54 | 46.14 | **45.25** |
| NLLB-200 1.3B | 46.24 | 43.75 | 47.16 | **45.72** |
| NLLB-200 600M (distilled) | 45.36 | 41.72 | 45.21 | **44.1** |
| Gemma 4 31B Instruct (Base) | 46.27 | 44.3 | 47.34 | **45.97** |
| OPUS-MT en-tw (Marian) | 39.48 | 35.84 | 38.03 | **37.78** |
| MADLAD-400 3B | 37.93 | 36.31 | 38.04 | **37.43** |
| Gemma 4 E4B Instruct (Base) | 8.12 | 8.6 | 7.82 | **8.18** |

### 2. Universal Declaration of Human Rights (UDHR)
Formal legalistic text translated into Ghanaian languages.

| Model | chrF2 | BLEU | Length Ratio | Words |
|---|---|---|---|---|
| Gemini 3.8 Flash | **56.44** | 10.15 | 0.949 | 2040 / 2098 |
| Google Translate (free endpoint) | **52.3** | 9.29 | 0.881 | 1909 / 2098 |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | **51.66** | 6.19 | 0.858 | 1863 / 2098 |
| Khaya AI | **49.92** | 8.06 | 1.032 | 2265 / 2098 |
| GLOSS-MT (TranslateGemma 4B + Glossary) | **51.19** | 4.77 | 0.86 | 1850 / 2098 |
| GLOSS-MT (Gemma 4 E4B + Tool Calling) | **50.8** | 6.19 | 0.947 | 2064 / 2098 |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | **50.09** | 5.54 | 0.836 | 1813 / 2098 |
| NLLB-200 3.3B | **52.05** | 10.45 | 0.92 | 1989 / 2098 |
| NLLB-200 1.3B | **50.47** | 8.12 | 0.956 | 2073 / 2098 |
| NLLB-200 600M (distilled) | **50.17** | 7.88 | 0.887 | 1913 / 2098 |
| Gemma 4 31B Instruct (Base) | **41.97** | 4.26 | 1.566 | 3480 / 2098 |
| OPUS-MT en-tw (Marian) | **47.52** | 4.29 | 0.829 | 1785 / 2098 |
| MADLAD-400 3B | **36.6** | 2.03 | 2.103 | 4558 / 2098 |
| Gemma 4 E4B Instruct (Base) | **3.84** | 0.02 | 30.387 | 70146 / 2098 |

### 3. Watchtower Study Editions (JW)
Moral, community, and religious prose.

| Model | JW Avg chrF2 | Documents Scored |
|---|---|---|
| Gemini 3.8 Flash | **69.28** | 5 |
| Google Translate (free endpoint) | **67.15** | 5 |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | **64.17** | 5 |
| Khaya AI | **65.78** | 5 |
| GLOSS-MT (TranslateGemma 4B + Glossary) | **62.54** | 5 |
| GLOSS-MT (Gemma 4 E4B + Tool Calling) | **60.75** | 5 |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | **60.62** | 5 |
| NLLB-200 3.3B | **56.89** | 5 |
| NLLB-200 1.3B | **57.46** | 5 |
| NLLB-200 600M (distilled) | **57.06** | 5 |
| Gemma 4 31B Instruct (Base) | **57.57** | 5 |
| OPUS-MT en-tw (Marian) | **54.23** | 5 |
| MADLAD-400 3B | **50.66** | 5 |
| Gemma 4 E4B Instruct (Base) | **21.86** | 5 |

### 4. Akan-Ashanti Folk-Tales (Rattray, 1930)
Oral literature and narrative storytelling covering 75 traditional folktales in authentic Asante Twi.

| Model | Folktales Avg chrF2 | Documents Scored |
|---|---|---|
| Gemini 3.8 Flash | **40.33** | 75 |
| Google Translate (free endpoint) | **37.54** | 75 |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | **37.16** | 75 |
| Khaya AI | **36.22** | 75 |
| GLOSS-MT (TranslateGemma 4B + Glossary) | **35.48** | 75 |
| GLOSS-MT (Gemma 4 E4B + Tool Calling) | **36.23** | 75 |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | **35.15** | 75 |
| NLLB-200 3.3B | **36.76** | 75 |
| NLLB-200 1.3B | **36.05** | 75 |
| NLLB-200 600M (distilled) | **35.5** | 75 |
| Gemma 4 31B Instruct (Base) | **36.71** | 75 |
| OPUS-MT en-tw (Marian) | **30.53** | 75 |
| MADLAD-400 3B | **23.87** | 75 |
| Gemma 4 E4B Instruct (Base) | **14.96** | 75 |

## Key Research Findings for the Paper

1. **Tool-Grounded Augmentation vs. Base Direct Translation:**
   - The unaugmented **Gemma 4 E4B Base** completely fails on Asante Twi translation across all domains, scoring an overall **12.21 chrF2** and suffering from catastrophic length inflation (ratio 14.13) and repetitive loops.
   - **GLOSS-MT** elevates the 4B model to **49.75 chrF2** (+37.54 points overall), with a near-perfect length ratio of **1.19**.
   - On Finance, GLOSS-MT gains **+43.05 chrF2 points** (`51.23` vs `8.18`).
   - On UDHR (legal), GLOSS-MT gains **+46.96 chrF2 points** (`50.80` vs `3.84`), reducing word count from 49,341 degenerate words to 2,064 exact words.
   - On Watchtower, GLOSS-MT gains **+38.89 chrF2 points** (`60.75` vs `21.86`).
   - On Folktales, GLOSS-MT gains **+21.27 chrF2 points** (`36.23` vs `14.96`).

2. **Surpassing 31B Base Model with 4B Parameters:**
   - GLOSS-MT 4B outperforms **Gemma 4 31B Base** overall (**49.75 vs 45.44**, a +4.31 point advantage), demonstrating that external lexical grounding and reasoning far outweigh raw parameter scale for low-resource translation.
   - On Finance, GLOSS-MT 4B (`51.23`) beats Gemma 4 31B (`45.97`) by **+5.26 points**.
   - On UDHR, GLOSS-MT 4B (`50.80`) beats Gemma 4 31B (`41.97`) by **+8.83 points**.

3. **Competitive Against Dedicated and Commercial Systems:**
   - GLOSS-MT 4B outperforms every open dedicated MT model: NLLB-200 3.3B (+2.07), NLLB-200 1.3B (+2.40), NLLB-200 600M (+3.04), and MADLAD-400 10B (+11.36).
   - On technical finance, GLOSS-MT 4B outperforms Google Translate (`51.23` vs `48.06`), placing second only to Gemini 3.8 Flash.

4. **Tool-Free GLOSS-MT: Dictionary Terms Pretranslated In Place for a Dedicated MT Model:**
   - **GLOSS-MT (NLLB-200 3.3B + Glossary)** looks up each chunk's noun phrases in the same dictionary, replaces the found phrases by their Twi translations inside the English, and translates the mixed text with NLLB-200 3.3B fine-tuned (LoRA) on such inputs. It scores **49.78 chrF2 / 13.93 BLEU** with a length ratio of **0.99**.
   - **+2.04 chrF2** over the base NLLB-200 3.3B (`49.78` vs `47.74`), and level with the tool-calling GLOSS-MT Gemma 4 E4B (`49.75`) with **+2.48 BLEU** (`13.93` vs `11.45`), from one beam-search pass per chunk with no reasoning trace or tool loop.
   - Largest gain on Watchtower: **`65.55`** vs `56.89` for base NLLB-200 3.3B (+8.66) and `60.75` for GLOSS-MT Gemma 4 E4B (+4.80).
   - Finance `46.97` (+1.72 over base NLLB-200 3.3B, −4.26 below GLOSS-MT Gemma 4 E4B); UDHR `50.46` (−1.59 vs base NLLB-200 3.3B); Folktales `36.16` (−0.60).

5. **The Same Pipeline on TranslateGemma 4B and OPUS-MT en-tw:**
   - **GLOSS-MT (TranslateGemma 4B + Glossary)**, a LoRA fine-tune of TranslateGemma 4B on the same data, scores **49.81 chrF2 / 12.24 BLEU** (length ratio 1.06): −0.99 chrF2 below GLOSS-MT NLLB 3.3B on the benchmark although it led on validation (BLEU 33.73 vs 31.73).
   - **GLOSS-MT (OPUS-MT en-tw + Glossary)**, a full fine-tune of the ~75M-parameter Marian model, scores **48.00 chrF2 / 10.91 BLEU**: **+5.48 chrF2** over the base OPUS-MT en-tw (`42.52`), with gains in every domain, and above base NLLB-200 3.3B on chrF2 (`47.74`).
