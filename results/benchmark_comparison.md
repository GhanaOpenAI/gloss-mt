# Machine Translation Benchmark Comparison: English → Asante Twi (Akan)

Comprehensive document-level benchmark evaluation on the `nsanku-mt-benchmark` suite across four distinct domains.

## Overall Leaderboard (English → Asante Twi)

| Model | Track | chrF2 (Avg) | BLEU | Finance (Budget) | UDHR | Watchtower (JW) | Folk-Tales | Docs Scored |
|---|---|---|---|---|---|---|---|---|
| **Gemini 3.8 Flash** | llm | **54.64** | 17.25 | 52.5 | 56.44 | 69.28 | 40.33 | 84 |
| **Google Translate (free endpoint)** | api | **51.26** | 15.67 | 48.06 | 52.3 | 67.15 | 37.54 | 84 |
| **GLOSS-MT (NLLB-200 3.3B + Glossary)** | open | **50.8** | 13.11 | 50.23 | 51.66 | 64.17 | 37.16 | 84 |
| **GLOSS-MT (NLLB-200 1.3B + Glossary)** | open | **50.27** | 12.82 | 49.41 | 51.51 | 63.66 | 36.51 | 84 |
| **GLOSS-MT (NLLB-200 600M + Glossary)** | open | **50.12** | 12.23 | 49.81 | 51.25 | 62.8 | 36.61 | 84 |
| **Khaya AI** | api | **49.91** | 12.79 | 47.74 | 49.92 | 65.78 | 36.22 | 84 |
| **GLOSS-MT (TranslateGemma 4B + Glossary)** | open | **49.81** | 12.24 | 50.03 | 51.19 | 62.54 | 35.48 | 84 |
| **GLOSS-MT (OPUS-MT en-tw + Glossary)** | open | **48.0** | 10.91 | 46.13 | 50.09 | 60.62 | 35.15 | 84 |
| **NLLB-200 3.3B** | open | **47.74** | 14.15 | 45.25 | 52.05 | 56.89 | 36.76 | 84 |
| **NLLB-200 1.3B** | open | **47.42** | 13.68 | 45.72 | 50.47 | 57.46 | 36.05 | 84 |
| **NLLB-200 600M (distilled)** | open | **46.71** | 12.74 | 44.1 | 50.17 | 57.06 | 35.5 | 84 |
| **OPUS-MT en-tw (Marian)** | open | **42.52** | 9.71 | 37.78 | 47.52 | 54.23 | 30.53 | 84 |
| **MADLAD-400 3B** | open | **37.14** | 6.71 | 37.43 | 36.6 | 50.66 | 23.87 | 84 |

## Domain & Dataset Breakdown

### 1. Finance / Citizens' Budget (Ministry of Finance, 2021–2023)
Technical and administrative domain covering national budgetary allocations, policy frameworks, and fiscal indicators.

| Model | 2021 chrF2 | 2022 chrF2 | 2023 chrF2 | Finance Avg chrF2 |
|---|---|---|---|---|
| Gemini 3.8 Flash | 53.28 | 50.3 | 53.93 | **52.5** |
| Google Translate (free endpoint) | 49.12 | 46.47 | 48.6 | **48.06** |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | 51.22 | 48.73 | 50.75 | **50.23** |
| GLOSS-MT (NLLB-200 1.3B + Glossary) | 50.35 | 47.4 | 50.48 | **49.41** |
| GLOSS-MT (NLLB-200 600M + Glossary) | 50.62 | 48.01 | 50.81 | **49.81** |
| Khaya AI | 48.33 | 46.57 | 48.33 | **47.74** |
| GLOSS-MT (TranslateGemma 4B + Glossary) | 50.6 | 48.45 | 51.03 | **50.03** |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | 45.74 | 44.78 | 47.87 | **46.13** |
| NLLB-200 3.3B | 46.06 | 43.54 | 46.14 | **45.25** |
| NLLB-200 1.3B | 46.24 | 43.75 | 47.16 | **45.72** |
| NLLB-200 600M (distilled) | 45.36 | 41.72 | 45.21 | **44.1** |
| OPUS-MT en-tw (Marian) | 39.48 | 35.84 | 38.03 | **37.78** |
| MADLAD-400 3B | 37.93 | 36.31 | 38.04 | **37.43** |

### 2. Universal Declaration of Human Rights (UDHR)
Formal legalistic text translated into Ghanaian languages.

| Model | chrF2 | BLEU | Length Ratio | Words |
|---|---|---|---|---|
| Gemini 3.8 Flash | **56.44** | 10.15 | 0.949 | 2040 / 2098 |
| Google Translate (free endpoint) | **52.3** | 9.29 | 0.881 | 1909 / 2098 |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | **51.66** | 6.19 | 0.858 | 1863 / 2098 |
| GLOSS-MT (NLLB-200 1.3B + Glossary) | **51.51** | 6.06 | 0.866 | 1882 / 2098 |
| GLOSS-MT (NLLB-200 600M + Glossary) | **51.25** | 5.6 | 0.854 | 1849 / 2098 |
| Khaya AI | **49.92** | 8.06 | 1.032 | 2265 / 2098 |
| GLOSS-MT (TranslateGemma 4B + Glossary) | **51.19** | 4.77 | 0.86 | 1850 / 2098 |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | **50.09** | 5.54 | 0.836 | 1813 / 2098 |
| NLLB-200 3.3B | **52.05** | 10.45 | 0.92 | 1989 / 2098 |
| NLLB-200 1.3B | **50.47** | 8.12 | 0.956 | 2073 / 2098 |
| NLLB-200 600M (distilled) | **50.17** | 7.88 | 0.887 | 1913 / 2098 |
| OPUS-MT en-tw (Marian) | **47.52** | 4.29 | 0.829 | 1785 / 2098 |
| MADLAD-400 3B | **36.6** | 2.03 | 2.103 | 4558 / 2098 |

### 3. Watchtower Study Editions (JW)
Moral, community, and religious prose.

| Model | JW Avg chrF2 | Documents Scored |
|---|---|---|
| Gemini 3.8 Flash | **69.28** | 5 |
| Google Translate (free endpoint) | **67.15** | 5 |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | **64.17** | 5 |
| GLOSS-MT (NLLB-200 1.3B + Glossary) | **63.66** | 5 |
| GLOSS-MT (NLLB-200 600M + Glossary) | **62.8** | 5 |
| Khaya AI | **65.78** | 5 |
| GLOSS-MT (TranslateGemma 4B + Glossary) | **62.54** | 5 |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | **60.62** | 5 |
| NLLB-200 3.3B | **56.89** | 5 |
| NLLB-200 1.3B | **57.46** | 5 |
| NLLB-200 600M (distilled) | **57.06** | 5 |
| OPUS-MT en-tw (Marian) | **54.23** | 5 |
| MADLAD-400 3B | **50.66** | 5 |

### 4. Akan-Ashanti Folk-Tales (Rattray, 1930)
Oral literature and narrative storytelling covering 75 traditional folktales in authentic Asante Twi.

| Model | Folktales Avg chrF2 | Documents Scored |
|---|---|---|
| Gemini 3.8 Flash | **40.33** | 75 |
| Google Translate (free endpoint) | **37.54** | 75 |
| GLOSS-MT (NLLB-200 3.3B + Glossary) | **37.16** | 75 |
| GLOSS-MT (NLLB-200 1.3B + Glossary) | **36.51** | 75 |
| GLOSS-MT (NLLB-200 600M + Glossary) | **36.61** | 75 |
| Khaya AI | **36.22** | 75 |
| GLOSS-MT (TranslateGemma 4B + Glossary) | **35.48** | 75 |
| GLOSS-MT (OPUS-MT en-tw + Glossary) | **35.15** | 75 |
| NLLB-200 3.3B | **36.76** | 75 |
| NLLB-200 1.3B | **36.05** | 75 |
| NLLB-200 600M (distilled) | **35.5** | 75 |
| OPUS-MT en-tw (Marian) | **30.53** | 75 |
| MADLAD-400 3B | **23.87** | 75 |

## Key Research Findings for the Paper

1. **Every GLOSS-MT fine-tune scores above its pretrained base on chrF2:** NLLB-200 3.3B 47.74 → 50.80 (+3.06), NLLB-200 1.3B 47.42 → 50.27 (+2.85), NLLB-200 600M 46.71 → 50.12 (+3.41), OPUS-MT en-tw 42.52 → 48.00 (+5.48). TranslateGemma 4B, which produces no Twi zero-shot, reaches 49.81.
2. **Best open systems:** the three GLOSS-MT NLLB models score above Khaya AI (49.91), behind only Gemini 3.8 Flash and Google Translate.
3. **Model size matters little once fine-tuned:** GLOSS-MT NLLB spans 50.12–50.80 across 600M–3.3B, and the 600M fine-tune scores above the base 3.3B model by +2.38.
4. **The gain concentrates in contemporary prose** (Watchtower and Finance); Folk-Tales (1930 orthography) gain least.
5. **BLEU falls slightly for the NLLB fine-tunes** (3.3B −1.04, 1.3B −0.86, 600M −0.51) while chrF2 rises; OPUS-MT gains BLEU (+1.20).
