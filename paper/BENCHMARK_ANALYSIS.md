# Empirical Evaluation and Domain-Specific Analysis

> **Research question.** Can we add controllability to machine translation, and reduce the burden of fine-tuning, without degrading the performance of pretrained MT models? (See the Central Research Question in `PREAMBLE.md`.)

**Evidence so far (English → Asante Twi, nsanku-mt-benchmark, chrF2):**

| Pretrained base | Base | GLOSS-MT fine-tune | Change | Fine-tune |
|---|:---:|:---:|:---:|---|
| NLLB-200 3.3B | 47.74 | 50.80 | **+3.06** | LoRA r=32 (2% of parameters), best at ~13 epochs |
| NLLB-200 1.3B | 47.42 | 50.27 | **+2.85** | full fine-tune, best at ~6 epochs |
| NLLB-200 600M (distilled) | 46.71 | 50.12 | **+3.41** | full fine-tune, best at ~28 epochs |
| OPUS-MT en-tw (75M) | 42.52 | 48.00 | **+5.48** | full fine-tune |
| TranslateGemma 4B | no Twi output zero-shot | 49.81 | n/a | LoRA r=32, best at ~3 epochs |
| MADLAD-400 3B | 37.14 | in training | | LoRA r=32 |

- **No degradation on chrF2, a small loss on BLEU:** every fine-tuned model scores above its pretrained base on chrF2, in every domain except the UDHR for NLLB 3.3B (−0.39). On BLEU the fine-tuned NLLB models are below their bases (3.3B −1.04, 1.3B −0.86, 600M −0.51), mostly on the UDHR and Watchtower, where the base output is shorter than the reference; OPUS-MT gains BLEU (+1.20).
- **Fine-tuning burden:** the fine-tune is done once, on a light budget; it is not repeated per domain.
- **Controllability, measured directly (Section 4):** the fine-tuned models write 79–97% of the dictionary terms inserted in the source (26–38% of the same terms appear without the dictionary), and when every entry is replaced by a different term, 70–97% of the new terms appear and about 2% of the old ones remain, with no retraining. The base models follow the dictionary much less (NLLB 3.3B: 66% on the Bible) or not at all (OPUS-MT: 2%). On human Bible references, translating with the dictionary scores at least as well in chrF2 as translating without it (NLLB 3.3B +0.3, OPUS-MT +2.8, TranslateGemma +5.0).

## 1. Disaggregating Macro Metrics: Addressing the Aggregation Fallacy

When evaluating translation systems across heterogeneous corpora, simple document-level unweighted macro averages can obscure critical qualitative differences across domains. 

In the `nsanku-mt-benchmark` suite (84 documents in total), the corpus distribution is heavily skewed:
- **Ministry of Finance (Citizens' Budget, 2021–2023)**: 3 comprehensive multi-page documents (1,310 paragraphs)
- **Universal Declaration of Human Rights (UDHR, 1948)**: 1 legal document (92 paragraphs)
- **Watchtower Study Editions (JW, 2026)**: 5 long-form publications (1,767 paragraphs)
- **Akan-Ashanti Folk-Tales (Rattray, 1930)**: 75 short narrative folktales (1,061 paragraphs)

Because **75 out of 84 documents (89.3%)** belong to a single historical storytelling dataset, an unweighted document-level metric reflects performance on 1930s folktales rather than general translation capability. The benchmark therefore averages per dataset first: each of the four datasets weighs the same in the reported mean.

---

## 2. Leaderboard (English → Asante Twi)

| Model | Track | Parameters | **chrF2 (Avg)** | **BLEU** | Finance | UDHR | Watchtower | Folktales | Docs |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemini 3.8 Flash** | Commercial LLM | Cloud | **54.64** | 17.25 | 52.50 | 56.44 | 69.28 | 40.33 | 84 |
| **Google Translate** | Commercial API | Cloud | **51.26** | 15.67 | 48.06 | 52.30 | 67.15 | 37.54 | 84 |
| **GLOSS-MT NLLB 3.3B (Ours)** | **Open MT + Dictionary Pretranslation** | **3.3B** | **50.80** | **13.11** | **50.23** | **51.66** | **64.17** | **37.16** | 84 |
| **GLOSS-MT NLLB 1.3B (Ours)** | **Open MT + Dictionary Pretranslation** | **1.3B** | **50.27** | **12.82** | **49.41** | **51.51** | **63.66** | **36.51** | 84 |
| **GLOSS-MT NLLB 600M (Ours)** | **Open MT + Dictionary Pretranslation** | **600M** | **50.12** | **12.23** | **49.81** | **51.25** | **62.80** | **36.61** | 84 |
| **Khaya AI** | Commercial API | Cloud | **49.91** | 12.79 | 47.74 | 49.92 | 65.78 | 36.22 | 84 |
| **GLOSS-MT TranslateGemma (Ours)** | **Translation LLM + Dictionary Pretranslation** | **4B** | **49.81** | **12.24** | **50.03** | **51.19** | **62.54** | **35.48** | 84 |
| **GLOSS-MT OPUS-MT (Ours)** | **Open MT + Dictionary Pretranslation** | **75M** | **48.00** | **10.91** | **46.13** | **50.09** | **60.62** | **35.15** | 84 |
| **NLLB-200 3.3B** | Dedicated Open MT | 3.3B | **47.74** | 14.15 | 45.25 | 52.05 | 56.89 | 36.76 | 84 |
| **NLLB-200 1.3B** | Dedicated Open MT | 1.3B | **47.42** | 13.68 | 45.72 | 50.47 | 57.46 | 36.05 | 84 |
| **NLLB-200 600M** | Distilled Open MT | 600M | **46.71** | 12.74 | 44.10 | 50.17 | 57.06 | 35.50 | 84 |
| **Gemma 4 31B Base** | Foundation LLM | 31B | **45.55** | 11.93 | 45.97 | 41.97 | 57.57 | 36.71 | 84 |
| **OPUS-MT en-tw** | Dedicated Open MT | 75M | **42.52** | 9.71 | 37.78 | 47.52 | 54.23 | 30.53 | 84 |
| **MADLAD-400 3B** | Dedicated Open MT | 3B | **37.14** | 6.71 | 37.43 | 36.60 | 50.66 | 23.87 | 84 |

Gemma 4 31B (prompted, no fine-tuning) is kept as a general-purpose LLM reference; the systems studied are dedicated translation models and a translation-specialised LLM.

### 2.1 Orthographic and Diachronic Divergence of the Folk-Tales
The Rattray (1930) corpus uses pre-standardization colonial Gold Coast orthography with archaic phonological transcriptions, dialectal Asante idioms, and non-standardized morpheme segmentations. 

Our dictionary and fine-tuning corpus are strictly aligned with modern standardized Asante Twi orthography (established by the Bureau of Ghana Languages post-1978). In contemporary domains, this alignment ensures high orthographic fidelity; however, when evaluated against a 1930 reference transcript, standard modern spelling incurs penalty against century-old colonial variants. This is why every system scores lowest on the Folk-Tales and why the fine-tunes gain least there.

---

## 3. GLOSS-MT: Dictionary Pretranslation for Dedicated Translation Models

### 3.1 Motivation
Encoder-decoder translation models (NLLB, OPUS-MT, MADLAD) and translation-specialised LLMs (TranslateGemma) take only source text in a fixed input format: there is no separate channel through which to give them a terminology list. GLOSS-MT therefore applies the offline dictionary *before* translation, inside the source text, and fine-tunes the model to use the terms it finds there.

### 3.2 Method
1. **Chunking.** Each document is split into lines, and consecutive lines are packed into chunks of at most 192 source tokens, cutting only at line boundaries so that English and Twi chunks stay aligned and markdown structure (headings, lists, tables) is preserved. A literal `"\n"` token is added to the tokenizer.
2. **Lookup.** Each chunk's noun phrases are extracted with spaCy and looked up in a dictionary with **exactly one translation per phrase**: the 412k entries that previously carried 2–3 candidates were re-translated into a single translation with Gemini 3.8 Flash, and every phrase aligned to a single Twi span in the parallel data (step 4) takes that span (613,259 entries).
3. **In-place pretranslation.** Every found phrase is replaced inside the English chunk by its Twi translation in angle brackets (whole words, longest phrase first, source casing kept); phrases not in the dictionary stay plain English. For example, *"Our people have courage, but courage without foundation is foolishness."* becomes *"Our &lt;nnipa&gt; have &lt;akokoduro&gt;, but &lt;akokoduro&gt; without &lt;fapem&gt; is &lt;nkwasiasɛm&gt;."* The model writes the terms in brackets too, and the pipeline removes them from its output.
4. **Fine-tuning.** Training inputs are built from the gold translations: Gemini aligns each English noun phrase of a chunk to its span in the reference Twi, and each aligned pair is bracketed on both sides, occurrence for occurrence (31,909 training chunks). NLLB-200 3.3B is fine-tuned with LoRA (r=32 on all attention and feed-forward projections plus the new-line embedding; 69M trainable parameters, 2%) to translate these mixed inputs into the bracketed gold Twi. Every 1,000 steps the model translates 200 fixed validation chunks; the checkpoint with the best mean of BLEU, chrF2 and 100·e^(−loss) is kept, with early stopping. The best checkpoint came at step 13,000 (~13 epochs; validation chrF2 63.66, BLEU 31.73).

The chunked, pretranslated data is released as [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi) and the model as [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b).

### 3.3 Results: NLLB-200 3.3B (English → Asante Twi, chrF2)

| Domain / Corpus | Documents | NLLB-200 3.3B (base) | GLOSS-MT NLLB 3.3B | Gain vs base NLLB |
|---|:---:|:---:|:---:|:---:|
| **Citizens' Budget (Finance)** | 3 | 45.25 | **50.23** | +4.98 |
| **UDHR (Human Rights / Legal)** | 1 | **52.05** | 51.66 | −0.39 |
| **Watchtower (JW / Religious Prose)** | 5 | 56.89 | **64.17** | **+7.28** |
| **Akan Folk-Tales (Rattray, 1930)** | 75 | 36.76 | **37.16** | +0.40 |
| **Mean of datasets (chrF2)** | 84 | 47.74 | **50.80** | **+3.06** |
| **BLEU** | 84 | **14.15** | 13.11 | −1.04 |
| **Length ratio** | 84 | 1.01 | **1.00** | |

- **Best open system.** GLOSS-MT NLLB reaches 50.80 chrF2, above Khaya AI (49.91) and behind only Gemini 3.8 Flash and Google Translate, from one beam-search pass per chunk.
- **The gain concentrates in contemporary prose.** Watchtower rises by +7.28 and Finance by +4.98 over base NLLB-200 3.3B.
- **Gold-aligned training data helps.** The previous release, fine-tuned on inputs pretranslated with the dictionary rather than aligned to the references, reached 49.78 chrF2 (Finance 46.97, UDHR 50.46, Watchtower 65.55, Folk-Tales 36.16). Training on gold-aligned brackets raises Finance by +3.26, UDHR by +1.20 and Folk-Tales by +1.00, at a cost of −1.38 on Watchtower and −0.82 BLEU.
- **Short legal output.** On the UDHR the model's output is 14% shorter than the reference (length ratio 0.86, base NLLB 0.92), so some content may be dropped.
- **Dictionary terms act as hints, not constraints.** The model often keeps institution names in English (*"Ministry of Finance"*) or prefers its own synonym over a pretranslated term, as the human references in the training data do; terminology control is therefore softer than an explicit constraint. Section 4 measures how often the terms are followed (88–94% for this model).

### 3.4 Qualitative Note: Table-of-Contents Dot Leaders
The Citizens' Budget documents open with tables of contents (*"CONCLUSION ......... 23"*). The fine-tuned model, which saw no dot leaders in training, looped on them and emitted roughly ten times the source's dots (16,302 vs 1,695 characters on the 2021 budget), occasionally followed by invented text. The pipeline now keeps dot leaders and page references out of the model and restores them unchanged. On the 2021 budget this raised chrF2 from 45.63 to 47.82 and BLEU from 9.37 to 17.57. All reported numbers use this pipeline.

### 3.5 Other Base Models: TranslateGemma 4B and OPUS-MT en-tw
The same data and pipeline were used with two more base models:
- **TranslateGemma 4B** (translation-specialised Gemma 3 LLM), LoRA r=32 (60M trainable parameters). It has no Twi code, so the target is Akan (`ak`); zero-shot it answered with non-Twi text (chrF2 4.15 on the GLOSS-MT test split). Training was stopped after step 5,000 and the best checkpoint (step 3,000, ~3 epochs) kept: on the same 200 validation chunks it reached BLEU 33.73 / chrF2 64.96, above GLOSS-MT NLLB 3.3B (31.73 / 63.66). Decoding is greedy.
- **OPUS-MT en-tw** (Helsinki-NLP MarianMT, ~75M parameters), full fine-tune. Its length limit is 512 tokens, so chunks are split at line boundaries into pieces that fit (27,596 training pieces from 31,909 chunks), and inference chunks are 128 source tokens. It trained for the full 30 epochs; the best checkpoint (step 23,000) reached BLEU 27.67 / chrF2 60.87 on 200 validation pieces (not the same set as above), with the score flat over the last ~10 epochs.

| Domain / Corpus | Documents | GLOSS-MT NLLB 3.3B | GLOSS-MT TranslateGemma 4B | OPUS-MT en-tw (base) | GLOSS-MT OPUS-MT | Gain vs base OPUS-MT |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Citizens' Budget (Finance)** | 3 | **50.23** | 50.03 | 37.78 | 46.13 | +8.35 |
| **UDHR (Human Rights / Legal)** | 1 | **51.66** | 51.19 | 47.52 | 50.09 | +2.57 |
| **Watchtower (JW / Religious Prose)** | 5 | **64.17** | 62.54 | 54.23 | 60.62 | +6.39 |
| **Akan Folk-Tales (Rattray, 1930)** | 75 | **37.16** | 35.48 | 30.53 | 35.15 | +4.62 |
| **Mean of datasets (chrF2)** | 84 | **50.80** | 49.81 | 42.52 | 48.00 | **+5.48** |
| **BLEU** | 84 | **13.11** | 12.24 | 9.71 | 10.91 | +1.20 |
| **Length ratio** | 84 | **1.00** | 1.06 | 0.86 | 0.97 | |

- **Validation lead does not transfer.** TranslateGemma beat NLLB 3.3B on validation but trails it by −0.99 chrF2 on the benchmark, in every domain. Its output runs long (length ratio 1.06; 1.55 on Folk-Tales), which the benchmark's document-level scoring penalises.
- **Large gain for a small model.** The pipeline raises OPUS-MT by +5.48 chrF2, most on Finance (+8.35), and the 75M-parameter fine-tune scores above the base NLLB-200 3.3B on chrF2 (48.00 vs 47.74), though not on BLEU.
- **Capacity still matters.** With the same data and dictionary, the ranking follows model size among dedicated MT models: NLLB 3.3B (50.80) > OPUS-MT 75M (48.00).

### 3.6 NLLB-200 1.3B
NLLB-200 1.3B was fully fine-tuned (Adafactor, learning rate 1e-4, batch 16 × 2) on the same chunks as NLLB 3.3B. The best checkpoint came at step 6,000 (~6 epochs; validation BLEU 30.20 / chrF2 62.14 on the same 200 chunks as NLLB 3.3B); BLEU and chrF2 rose slightly afterwards while the validation loss rose, and early stopping ended training at step 11,000.

| Domain / Corpus | Documents | NLLB-200 1.3B (base) | GLOSS-MT NLLB 1.3B | Gain vs base | GLOSS-MT NLLB 3.3B |
|---|:---:|:---:|:---:|:---:|:---:|
| **Citizens' Budget (Finance)** | 3 | 45.72 | 49.41 | +3.69 | **50.23** |
| **UDHR (Human Rights / Legal)** | 1 | 50.47 | 51.51 | +1.04 | **51.66** |
| **Watchtower (JW / Religious Prose)** | 5 | 57.46 | 63.66 | **+6.20** | **64.17** |
| **Akan Folk-Tales (Rattray, 1930)** | 75 | 36.05 | 36.51 | +0.46 | **37.16** |
| **Mean of datasets (chrF2)** | 84 | 47.42 | 50.27 | **+2.85** | **50.80** |
| **BLEU** | 84 | **13.68** | 12.82 | −0.86 | 13.11 |
| **Length ratio** | 84 | 0.96 | 0.99 | | 1.00 |

- **Gains in every domain.** Unlike NLLB 3.3B, the 1.3B fine-tune improves on its base on all four datasets, again most on Watchtower (+6.20) and Finance (+3.69). Base NLLB 1.3B drops content on Watchtower (length ratio 0.70); the fine-tune brings it to 0.90.
- **Close to the larger model.** At 40% of the parameters, it scores 0.53 chrF2 below GLOSS-MT NLLB 3.3B and above Khaya AI (49.91).
- **BLEU falls, as for NLLB 3.3B** (UDHR 8.12 → 6.06, Watchtower 25.49 → 23.70), while chrF2 rises on both: the fine-tune matches more of the reference's characters but fewer of its exact word sequences.

### 3.7 NLLB-200 600M
NLLB-200 distilled 600M was fully fine-tuned with the same settings as NLLB 1.3B. Unlike the 1.3B, it kept improving on validation for the whole 30-epoch schedule, flattening over the last ~8 epochs; the best checkpoint came at step 28,000 (~28 epochs; validation BLEU 32.41 / chrF2 63.99, above the 1.3B's 30.20 / 62.14).

| Domain / Corpus | Documents | NLLB-200 600M (base) | GLOSS-MT NLLB 600M | Gain vs base | GLOSS-MT NLLB 1.3B | GLOSS-MT NLLB 3.3B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Citizens' Budget (Finance)** | 3 | 44.10 | 49.81 | **+5.71** | 49.41 | **50.23** |
| **UDHR (Human Rights / Legal)** | 1 | 50.17 | 51.25 | +1.08 | 51.51 | **51.66** |
| **Watchtower (JW / Religious Prose)** | 5 | 57.06 | 62.80 | **+5.74** | 63.66 | **64.17** |
| **Akan Folk-Tales (Rattray, 1930)** | 75 | 35.50 | 36.61 | +1.11 | 36.51 | **37.16** |
| **Mean of datasets (chrF2)** | 84 | 46.71 | 50.12 | **+3.41** | 50.27 | **50.80** |
| **BLEU** | 84 | 12.74 | 12.23 | −0.51 | 12.82 | **13.11** |
| **Length ratio** | 84 | 0.92 | 0.99 | | 0.99 | 1.00 |

- **The largest gain of the NLLB family.** The 600M fine-tune gains +3.41 chrF2 over its base (1.3B +2.85, 3.3B +3.06), in every domain, and comes within 0.15 of the 1.3B fine-tune at under half its size. On Finance it scores above the 1.3B (49.81 vs 49.41).
- **The size series is flat once fine-tuned.** Base NLLB spans 46.71–47.74 across the three sizes and GLOSS-MT NLLB spans 50.12–50.80: the dictionary pipeline lifts each size by about 3 chrF2, and a 600M model with the dictionary scores above the base 3.3B model by +2.38.
- **Same BLEU pattern.** BLEU falls slightly (−0.51), on the UDHR (7.88 → 5.60) and Watchtower (24.26 → 21.61), while chrF2 rises on both.
- **Higher validation scores did not carry over.** The 600M scored above the 1.3B on validation (chrF2 63.99 vs 62.14) but slightly below it on the benchmark (50.12 vs 50.27), as TranslateGemma did against NLLB 3.3B (Section 3.5).

### 3.8 Next: MADLAD-400 3B
The same recipe is being applied to MADLAD-400 3B (LoRA r=32, target token `<2ak>`), to test a third architecture. Its first validation evaluation (step 1,000) reached BLEU 21.69 / chrF2 45.83, so the shared embeddings load correctly.

---

## 4. Controllability: Does the Output Follow the Dictionary?

### 4.1 Method
The benchmark shows translation quality, not whether the dictionary controls the output. `training/eval_controllability.py` measures this directly. Each test set is translated three times by the same model, changing only the dictionary:
- **dict:** the GLOSS-MT dictionary, as deployed.
- **none:** no dictionary. This shows how often the model produces the same Twi terms by itself.
- **swap:** a counterfactual dictionary. Every dictionary phrase in the test set gets a different real Twi term, taken from the other terms of the set, that occurs neither in the source nor in the dict and none outputs for that passage. A swapped term can therefore only reach the output through the dictionary. The swapped terms are usually wrong for the phrase: the test measures steering, not quality.

A term counts as used when its words occur in order as whole words of the output (case-insensitive). The metrics are:
- **adherence:** share of inserted terms that appear in the output.
- **natural rate:** share of the same terms that appear without the dictionary.
- **control gain:** adherence − natural rate, i.e. what the dictionary adds.
- **swap adherence:** share of swapped-in terms that appear in the swap output.
- **swap leak:** share of the original terms still present after the swap.

Two test sets are used:
- **Validation:** 300 chunks of the GLOSS-MT validation split (5,387 inserted terms). Its references come from the GLOSS-MT parallel data, which is not entirely human-curated.
- **Asante Twi Bible:** 300 passages of three consecutive verses (2,766 inserted terms; `scripts/build_bible_testset.py`). English is the Good News Translation; Twi is the human Asante Twi Nkwa Asɛm, both from the africa-corpus Bible collection. No passage shares an 8-word span with the training data. Both versions are free, meaning-based translations, so absolute scores are low and the comparison that matters is with vs without the dictionary. Only 35.9% of the inserted dictionary terms occur in the Bible reference (48.5% on validation), so the dictionary often disagrees with the human translator.

### 4.2 Results

| Model | Test set | Adherence | Natural rate | Control gain | Swap adherence | Swap leak | chrF2 dict / none | BLEU dict / none |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GLOSS-MT TranslateGemma 4B | Bible | **96.6** | 26.2 | **+70.4** | **97.3** | 1.9 | 37.15 / 32.13 | 6.36 / 4.69 |
| GLOSS-MT NLLB 3.3B | Bible | 94.4 | 38.4 | +56.0 | 95.2 | 2.2 | **41.21** / 40.92 | 10.75 / 10.91 |
| GLOSS-MT OPUS-MT | Bible | 79.2 | 32.5 | +46.7 | 69.7 | 1.7 | 36.37 / 33.56 | 6.55 / 5.92 |
| NLLB-200 3.3B (base) | Bible | 66.0 | 33.9 | +32.1 | 72.0 | 2.8 | 38.94 / 39.58 | 10.07 / **12.27** |
| OPUS-MT en-tw (base) | Bible | 1.9 | 30.3 | −28.4 | 0.4 | 1.4 | 19.13 / 30.13 | 1.98 / 4.93 |
| GLOSS-MT TranslateGemma 4B | Validation | **97.8** | 36.1 | **+61.8** | **95.9** | 6.1 | 65.98 / 67.66 | 43.59 / 44.29 |
| GLOSS-MT NLLB 3.3B | Validation | 88.4 | 37.8 | +50.6 | 75.8 | 6.6 | 65.36 / **71.00** | 42.72 / **49.23** |
| GLOSS-MT OPUS-MT | Validation | 76.2 | 34.7 | +41.4 | 61.0 | 5.6 | 59.17 / 64.31 | 35.22 / 41.24 |
| NLLB-200 3.3B (base) | Validation | 55.8 | 26.9 | +28.9 | 49.0 | 3.6 | 41.32 / 45.51 | 19.34 / 22.93 |
| OPUS-MT en-tw (base) | Validation | 1.8 | 13.9 | −12.1 | 0.1 | 1.3 | 15.02 / 26.05 | 2.63 / 6.99 |

TranslateGemma 4B has no base row: zero-shot it does not produce Twi (Section 3.5).

### 4.3 Findings
- **The dictionary controls the fine-tuned models.** On the Bible, GLOSS-MT TranslateGemma and NLLB 3.3B write 95–97% of the inserted terms. After every entry is replaced, they write 95–97% of the new terms, and only about 2% of the original terms survive. Editing the dictionary therefore changes the translation as intended, without retraining. In one Bible passage, swapping *Jerusalem* and *David* for unrelated terms changed every occurrence in the output.
- **Fine-tuning creates the control.** Base OPUS-MT ignores the bracketed terms (1.9% adherence) and its translation degrades when they are present (chrF2 30.13 → 19.13). Base NLLB 3.3B copies some Twi terms from its input (66.0% on the Bible, 55.8% on validation), and fine-tuning raises this to 94.4% and 88.4%.
- **Control strength varies by model.** TranslateGemma follows multi-word terms as reliably as single words (98.1% vs 97.7% on validation). OPUS-MT, the smallest model, follows 70% of swapped terms on the Bible and 61% on validation. The swapped terms are often wrong for their context, and a small model is more likely to drop them.
- **No quality cost on human references.** On the Bible, translating with the dictionary scores at least as well in chrF2 as without it: TranslateGemma +5.02, OPUS-MT +2.81, NLLB 3.3B +0.29 (BLEU −0.16). The fine-tuned NLLB 3.3B with the dictionary scores above base NLLB 3.3B used normally on chrF2 (41.21 vs 39.58) but below it on BLEU (10.75 vs 12.27), matching the BLEU loss on the benchmark (Section 3.3).
- **The validation set gives the opposite result.** There, the dictionary lowers chrF2 for every fine-tuned model (NLLB 3.3B −5.64, OPUS-MT −5.14, TranslateGemma −1.68). The validation references are not all human-curated, and the fine-tuned models score higher on them without the dictionary than with it. The Bible results suggest the drop comes from these references, not from a real loss of quality, which is why the human-referenced set is used for this comparison.

### 4.4 Limits
- Term matching is by exact whole words. A term that Twi morphology changes in the output (a prefix or plural form) counts as not used, so adherence is a lower bound.
- The swap test uses wrong terms on purpose, so swap adherence measures steering, not the quality of the swapped translations.
- The Bible is one domain, with free translations on both sides. Further human-referenced sets in other domains would strengthen the no-degradation conclusion.
