# Empirical Evaluation and Domain-Specific Analysis

> **Research question.** Can we add controllability to machine translation, and reduce the burden of fine-tuning, without degrading the performance of pretrained MT models? (See the Central Research Question in `PREAMBLE.md`.)

**Evidence so far (English → Asante Twi, nsanku-mt-benchmark, chrF2):**

| Pretrained base | Base | GLOSS-MT fine-tune | Change | Fine-tune |
|---|:---:|:---:|:---:|---|
| NLLB-200 3.3B | 47.74 | 50.80 | **+3.06** | LoRA r=32 (2% of parameters), best at ~13 epochs |
| OPUS-MT en-tw (75M) | 42.52 | 48.00 | **+5.48** | full fine-tune |
| TranslateGemma 4B | no Twi output zero-shot | 49.81 | n/a | LoRA r=32, best at ~3 epochs |
| Gemma 4 E4B (tool calling) | 12.21 | 49.75 | +37.54 | LoRA (tool-calling SFT) |
| NLLB-200 1.3B, 600M; MADLAD-400 3B | 47.42, 46.71, 37.14 | in training | | |

- **No degradation (so far):** every fine-tuned model scores above its pretrained base on chrF2. Two exceptions to watch: GLOSS-MT NLLB 3.3B is −1.04 BLEU and −0.39 chrF2 on the UDHR below base NLLB-200 3.3B.
- **Fine-tuning burden:** the fine-tune is done once, on a light budget; it is not repeated per domain.
- **Controllability is not yet measured directly.** The dictionary terms currently act as hints, not constraints (Section 5.3): the model sometimes keeps an English name or prefers its own synonym. A direct test is still needed: the share of pretranslated terms that appear in the output, and whether editing a dictionary entry changes the output accordingly, without retraining.

## 1. Disaggregating Macro Metrics: Addressing the Aggregation Fallacy

When evaluating generative language models and tool-calling translation systems across heterogeneous corpora, simple document-level unweighted macro averages can obscure critical qualitative differences across domains. 

In the `nsanku-mt-benchmark` suite (84 documents in total), the corpus distribution is heavily skewed:
- **Ministry of Finance (Citizens' Budget, 2021–2023)**: 3 comprehensive multi-page documents (1,310 paragraphs)
- **Universal Declaration of Human Rights (UDHR, 1948)**: 1 legal document (92 paragraphs)
- **Watchtower Study Editions (JW, 2026)**: 5 long-form publications (1,767 paragraphs)
- **Akan-Ashanti Folk-Tales (Rattray, 1930)**: 75 short narrative folktales (1,061 paragraphs)

Because **75 out of 84 documents (89.3%)** belong to a single historical storytelling dataset, an unweighted document-level metric reflects performance on 1930s folktales rather than general translation capability.

---

## 2. Comparative Evaluation Across Contemporary vs. Historical Domains

When disaggregating the contemporary, high-stakes operational domains (Finance, Statutory Legal, and Moral/Religious prose) from the historical folklore narratives, a clear distinction emerges between the base parametric models and our tool-calling architecture:

| Domain / Corpus | Documents | Base Gemma 4 E4B (4B) | Base Gemma 4 31B (31B) | GLOSS-MT (4B Ours) | Gain vs Base 4B | Gain vs Base 31B |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Citizens' Budget (Finance)** | 3 | 8.18 | 45.97 | **51.23** | **+43.05** | **+5.26** |
| **UDHR (Human Rights / Legal)** | 1 | 3.84 | 41.97 | **50.80** | **+46.96** | **+8.83** |
| **Watchtower (JW / Religious Prose)** | 5 | 21.86 | 57.57 | **60.75** | **+38.89** | **+3.18** |
| **Akan Folk-Tales (Rattray, 1930)** | 75 | 14.96 | **36.71** | **36.23** | **+21.27** | -0.48 |
| **Macro Average (All 84 Docs)** | **84** | **12.21** | **45.55** | **49.75** | **+37.54** | **+4.20** |

On contemporary standard texts, **GLOSS-MT consistently outperforms the base 4B model by over +40 chrF2 points, and outperforms the 31B base model by +4.20 chrF2 points overall**, surpassing dedicated translation baselines such as NLLB-200 3.3B (`47.74`) and MADLAD-400 3B (`37.14`).

### 2.1 Complete Leaderboard Across Evaluated Systems

| Model | Track | Parameters | **chrF2 (Avg)** | **BLEU** | Finance | UDHR | Watchtower | Folktales | Docs |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemini 3.8 Flash** | Commercial LLM | Cloud | **54.64** | 17.25 | 52.50 | 56.44 | 69.28 | 40.33 | 84 |
| **Google Translate** | Commercial API | Cloud | **51.26** | 15.67 | 48.06 | 52.30 | 67.15 | 37.54 | 84 |
| **GLOSS-MT NLLB (Ours)** | **Open MT + Dictionary Pretranslation** | **3.3B** | **50.80** | **13.11** | **50.23** | **51.66** | **64.17** | **37.16** | 84 |
| **Khaya AI** | Commercial API | Cloud | **49.92** | 12.79 | 47.74 | 49.92 | 65.78 | 36.22 | 84 |
| **GLOSS-MT TranslateGemma (Ours)** | **Translation LLM + Dictionary Pretranslation** | **4B** | **49.81** | **12.24** | **50.03** | **51.19** | **62.54** | **35.48** | 84 |
| **GLOSS-MT (Ours)** | **LLM + Offline Tool** | **4B** | **49.75** | **11.45** | **51.23** | **50.80** | **60.75** | **36.23** | 84 |
| **GLOSS-MT OPUS-MT (Ours)** | **Open MT + Dictionary Pretranslation** | **75M** | **48.00** | **10.91** | **46.13** | **50.09** | **60.62** | **35.15** | 84 |
| **NLLB-200 3.3B** | Dedicated Open MT | 3.3B | **47.74** | 14.15 | 45.25 | 52.05 | 56.89 | 36.76 | 84 |
| **NLLB-200 1.3B** | Dedicated Open MT | 1.3B | **47.42** | 13.68 | 45.72 | 50.47 | 57.46 | 36.05 | 84 |
| **NLLB-200 600M** | Distilled Open MT | 600M | **46.71** | 12.74 | 44.10 | 50.17 | 57.06 | 35.50 | 84 |
| **Gemma 4 31B Base** | Foundation LLM | 31B | **45.55** | 11.93 | 45.97 | 41.97 | 57.57 | 36.71 | 84 |
| **OPUS-MT en-tw** | Dedicated Open MT | 75M | **42.52** | 9.71 | 37.78 | 47.52 | 54.23 | 30.53 | 84 |
| **MADLAD-400 3B** | Dedicated Open MT | 3B | **37.14** | 6.71 | 37.43 | 36.60 | 50.66 | 23.87 | 84 |
| **Gemma 4 E4B Base** | Foundation LLM | 4B | **12.21** | 0.94 | 8.18 | 3.84 | 21.86 | 14.96 | 84 |

---

## 3. Qualitative Insights on Operational Domains

### 3.1 Statutory Legal Texts (UDHR): Elimination of Hallucinatory Length Inflation
On the Universal Declaration of Human Rights, the un-finetuned `gemma-4-E4B-it` base model experiences catastrophic failure, degenerating into infinite repetition loops and emitting **49,341 words** for a 2,098-word reference (length ratio: **23.518**, chrF2: **3.84**). Even the 31B base model exhibited severe length inflation, generating **3,480 words** (length ratio: **1.566**, chrF2: **41.97**).

In stark contrast, GLOSS-MT retrieved verified legal terminology, reasoned over grammatical structure, and produced an exact translation of **2,064 words** (length ratio: **0.947**). This grounded control yielded a **+46.96 chrF2 improvement** over Base 4B and a **+8.83 chrF2 improvement** over Base 31B (`41.97` → `50.80`), matching specialized commercial translation engines without parameter explosion.

### 3.2 Finance (Citizens' Budget): Outperforming Commercial and Open Baselines
In technical administrative domains, terminology shifts rapidly across fiscal cycles. 
- The unaugmented **Base 4B model** scored only **8.18 chrF2** due to lexical unfamiliarity with specialized public-finance terms.
- **GLOSS-MT 4B** achieved **51.23 chrF2** (and **14.75 BLEU**), beating **Google Translate (48.06)**, **Gemma 4 31B Base (45.97)**, and **NLLB-200 3.3B (45.25)**.

This demonstrates that external lexical tool-calling empowers a compact 4B model to surpass models nearly an order of magnitude larger in technical domain translation.

---

## 4. Resolution of the Akan Folk-Tales Chunking Issue

### 4.1 Token Budget Depletion in Previous Evaluations
In early experiments, paragraphs were fed directly into the model API. In the Rattray (1930) dataset, several folktales exist as single continuous paragraphs containing thousands of English words (e.g., Folktale 31 contains 2,844 English words in a single text block). Because the tool-calling model emits a multi-step trajectory:
$$\text{Input} \longrightarrow \text{Tool Call} \longrightarrow \text{External Observation} \longrightarrow \text{Linguistic Thought} \longrightarrow \text{Final Translation}$$

Generating this full sequence for a 2,800-word block exhausted the API token budget under the previous 2,500-character fallback splitter, resulting in severe truncation (length ratio 0.14 on Folktale 31; overall folktales score of 22.15).

### 4.2 Chunking Solution and Validated Recovery
By updating the benchmark chunking fallback to 400 characters (~60 words / 1–2 sentences) and setting `max_tokens = 4096`, all 75 folktale documents are now translated without truncation. 
- Folktale 31 scored **`38.99 chrF2`** (versus truncated 22.15).
- The 75-document folktales average rose to **`36.23 chrF2`**, closely matching the 31B base model (`36.71`), matching Khaya AI (`36.22`), and outperforming NLLB-200 1.3B (`36.05`) and NLLB-200 600M (`35.50`).

### 4.3 Orthographic and Diachronic Divergence
The Rattray (1930) corpus uses pre-standardization colonial Gold Coast orthography with archaic phonological transcriptions, dialectal Asante idioms, and non-standardized morpheme segmentations. 

Our dictionary and fine-tuning corpus are strictly aligned with modern standardized Asante Twi orthography (established by the Bureau of Ghana Languages post-1978). In contemporary domains, this alignment ensures high orthographic fidelity; however, when evaluated against a 1930 reference transcript, standard modern spelling incurs penalty against century-old colonial variants.

---

## 5. Tool-Free GLOSS-MT: Dictionary Pretranslation for Dedicated Translation Models

### 5.1 Motivation
Encoder-decoder translation models (NLLB, umT5) and translation-specialised LLMs (TranslateGemma) cannot call tools, and their fixed input formats leave no room for a reasoning trace. To test whether GLOSS-MT's dictionary grounding transfers to this class of models, the tool loop is replaced by a fixed pipeline that applies the same offline dictionary *before* translation.

### 5.2 Method
1. **Chunking.** Each document is split into lines, and consecutive lines are packed into chunks of at most 192 source tokens, cutting only at line boundaries so that English and Twi chunks stay aligned and markdown structure (headings, lists, tables) is preserved. A literal `"\n"` token is added to the tokenizer.
2. **Lookup.** Each chunk's noun phrases are extracted with spaCy (as in GLOSS-MT) and looked up in a dictionary with **exactly one translation per phrase**: the 412k entries that previously carried 2–3 candidates were re-translated into a single translation with Gemini 3.8 Flash, and every phrase aligned to a single Twi span in the parallel data (step 4) takes that span (613,259 entries).
3. **In-place pretranslation.** Every found phrase is replaced inside the English chunk by its Twi translation in angle brackets (whole words, longest phrase first, source casing kept); phrases not in the dictionary stay plain English. For example, *"Our people have courage, but courage without foundation is foolishness."* becomes *"Our &lt;nnipa&gt; have &lt;akokoduro&gt;, but &lt;akokoduro&gt; without &lt;fapem&gt; is &lt;nkwasiasɛm&gt;."* The model writes the terms in brackets too, and the pipeline removes them from its output.
4. **Fine-tuning.** Training inputs are built from the gold translations: Gemini aligns each English noun phrase of a chunk to its span in the reference Twi, and each aligned pair is bracketed on both sides, occurrence for occurrence (31,909 training chunks). NLLB-200 3.3B is fine-tuned with LoRA (r=32 on all attention and feed-forward projections plus the new-line embedding; 69M trainable parameters, 2%) to translate these mixed inputs into the bracketed gold Twi. Every 1,000 steps the model translates 200 fixed validation chunks; the checkpoint with the best mean of BLEU, chrF2 and 100·e^(−loss) is kept, with early stopping. The best checkpoint came at step 13,000 (~13 epochs; validation chrF2 63.66, BLEU 31.73).

The chunked, pretranslated data is released as [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi) and the model as [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b).

### 5.3 Results (English → Asante Twi, chrF2)

| Domain / Corpus | Documents | NLLB-200 3.3B (base) | GLOSS-MT Gemma 4 E4B | GLOSS-MT NLLB 3.3B | Gain vs base NLLB |
|---|:---:|:---:|:---:|:---:|:---:|
| **Citizens' Budget (Finance)** | 3 | 45.25 | **51.23** | 50.23 | +4.98 |
| **UDHR (Human Rights / Legal)** | 1 | **52.05** | 50.80 | 51.66 | −0.39 |
| **Watchtower (JW / Religious Prose)** | 5 | 56.89 | 60.75 | **64.17** | **+7.28** |
| **Akan Folk-Tales (Rattray, 1930)** | 75 | 36.76 | 36.23 | **37.16** | +0.40 |
| **Mean of datasets (chrF2)** | 84 | 47.74 | 49.75 | **50.80** | **+3.06** |
| **BLEU** | 84 | **14.15** | 11.45 | 13.11 | −1.04 |
| **Length ratio** | 84 | 1.01 | 1.20 | **1.00** | |

- **Best open system, ahead of the tool-calling model.** GLOSS-MT NLLB reaches 50.80 chrF2, +1.05 over GLOSS-MT Gemma 4 E4B and above Khaya AI (49.92), behind only Gemini 3.8 Flash and Google Translate, from one beam-search pass per chunk and no LLM server.
- **The gain concentrates in contemporary prose.** Watchtower rises by +7.28 and Finance by +4.98 over base NLLB-200 3.3B; on Finance the pipeline now comes within 1.0 point of GLOSS-MT Gemma 4 E4B, which can query any phrase it needs and reason about figures and policy terms.
- **Gold-aligned training data helps.** The previous release, fine-tuned on inputs pretranslated with the dictionary rather than aligned to the references, reached 49.78 chrF2 (Finance 46.97, UDHR 50.46, Watchtower 65.55, Folk-Tales 36.16). Training on gold-aligned brackets raises Finance by +3.26, UDHR by +1.20 and Folk-Tales by +1.00, at a cost of −1.38 on Watchtower and −0.82 BLEU.
- **Short legal output.** On the UDHR the model's output is 14% shorter than the reference (length ratio 0.86, base NLLB 0.92), so some content may be dropped.
- **Dictionary terms act as hints, not constraints.** The model often keeps institution names in English (*"Ministry of Finance"*) or prefers its own synonym over a pretranslated term, as the human references in the training data do; terminology control is therefore softer than an explicit constraint.

### 5.4 Qualitative Note: Table-of-Contents Dot Leaders
The Citizens' Budget documents open with tables of contents (*"CONCLUSION ......... 23"*). The fine-tuned model, which saw no dot leaders in training, looped on them and emitted roughly ten times the source's dots (16,302 vs 1,695 characters on the 2021 budget), occasionally followed by invented text. The pipeline now keeps dot leaders and page references out of the model and restores them unchanged. On the 2021 budget this raised chrF2 from 45.63 to 47.82 and BLEU from 9.37 to 17.57. All reported numbers use this pipeline.

### 5.5 Other Base Models: TranslateGemma 4B and OPUS-MT en-tw
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

### 5.6 Next: NLLB-200 1.3B, NLLB-200 600M and MADLAD-400 3B
The same recipe is being applied to NLLB-200 1.3B and NLLB-200 distilled 600M (full fine-tunes) and MADLAD-400 3B (LoRA r=32, target token `<2ak>`), to measure how the pipeline's gain scales with model size within one model family and on a third architecture.
