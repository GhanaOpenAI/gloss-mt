---
language:
- en
- tw
license: cc-by-nc-4.0
tags:
- translation
- Akan
- Asante Twi
- nllb
- glossary
- terminology
- GLOSS-MT
base_model: facebook/nllb-200-distilled-600M
datasets:
- ghanaopenai/gloss-mt-parallel-en-twi
metrics:
- chrf
- bleu
pipeline_tag: translation
---

# GLOSS-MT (NLLB-200 600M + Glossary)

**`ghanaopenai/gloss-mt-nllb-600m`** is an English → Asante Twi (Akan) translation model: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M), fully fine-tuned to
translate text in which dictionary terms have already been **pretranslated in place**. An offline
English → Twi dictionary supplies the terms, so terminology can be changed by editing the
dictionary instead of retraining the model.

- **Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)
- **Supported by:** [Ghana NLP](https://ghananlp.org)
- **Code:** [GhanaOpenAI/gloss-mt](https://github.com/GhanaOpenAI/gloss-mt)
- **Live demo:** [Hugging Face Spaces: GLOSS-MT](https://huggingface.co/spaces/ghanaopenai/gloss-mt)
- **Training data:** [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi)
- **Dictionary:** 613,259 English → Twi phrases, one translation each (`twi_dictionary_single.tsv.gz`)

## How it works

```
English text
  │  split into line-aligned chunks of at most 192 tokens
  ▼
spaCy noun phrases ──► offline dictionary lookup
  │
  ▼
found phrases replaced by their Twi translation in angle brackets, in place
  "Our people have courage, but courage without foundation is foolishness."
  "Our <nnipa> have <akokoduro>, but <akokoduro> without <fapem> is <nkwasiasɛm>."
  │
  ▼
NLLB-200 600M  eng_Latn → twi_Latn (fine-tuned)  ──► Twi chunk with the terms in brackets
  │  brackets removed, chunks re-joined
  ▼
Twi text
```

Only phrases found in the dictionary are bracketed; the others stay plain English for the model
to translate. The model learned to keep each bracketed term and translate the sentence around it,
writing the term in brackets too, so the pipeline can strip them. Dot leaders of tables of
contents are kept out of the model and restored unchanged. Editing the dictionary changes the
terms the model writes, without retraining.

## Benchmark (nsanku-mt-benchmark, English → Asante Twi)

Document-level chrF2 on the [nsanku MT benchmark](https://github.com/GhanaNLP/nsanku-mt-benchmark)
(84 documents, 4 domains; a language score is the mean of its per-dataset scores):

| Model | chrF2 | BLEU | Length ratio | Finance (3) | Folktales (75) | Watchtower (5) | UDHR (1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemini 3.8 Flash | 54.64 | 17.25 | 1.02 | 52.50 | 40.33 | 69.28 | 56.44 |
| Google Translate | 51.26 | 15.67 | 1.01 | 48.06 | 37.54 | 67.15 | 52.30 |
| GLOSS-MT NLLB-200 3.3B | 50.80 | 13.11 | 1.00 | 50.23 | 37.16 | 64.17 | 51.66 |
| GLOSS-MT NLLB-200 1.3B | 50.27 | 12.82 | 0.99 | 49.41 | 36.51 | 63.66 | 51.51 |
| **GLOSS-MT NLLB-200 600M** | **50.12** | **12.23** | **0.99** | **49.81** | **36.61** | **62.80** | **51.25** |
| Khaya AI | 49.91 | 12.79 | 1.04 | 47.74 | 36.22 | 65.78 | 49.92 |
| GLOSS-MT TranslateGemma 4B | 49.81 | 12.24 | 1.06 | 50.03 | 35.48 | 62.54 | 51.19 |
| GLOSS-MT OPUS-MT en-tw | 48.00 | 10.91 | 0.97 | 46.13 | 35.15 | 60.62 | 50.09 |
| NLLB-200 600M (base) | 46.71 | 12.74 | 0.92 | 44.10 | 35.50 | 57.06 | 50.17 |

Fine-tuning plus the dictionary adds 3.41 chrF2 over the base NLLB-200 600M, in every domain (+5.71 on Finance, +5.74 on Watchtower), and scores 2.38 above the base NLLB-200 3.3B; BLEU is 0.51 lower than the base's.

## Training

- **Data:** 31,909 line-aligned chunks (≤ 192 tokens) of English–Twi documents. Each English noun
  phrase was aligned to its span in the gold Twi with Gemini, and the aligned phrases are bracketed
  on both sides (`english_pretranslated` → `twi_bracketed`), so the model learns to keep the
  bracketed terms.
- **Method:** full fine-tune, Adafactor, lr 1e-4, batch 16 × 2 gradient accumulation, bf16, 30 epochs,
  plus an added `"\n"` token that keeps markdown line structure.
- **Checkpoint choice:** every 1,000 steps the model translated 200 fixed validation chunks; the
  checkpoint with the best mean of BLEU, chrF2 and 100·e^(−loss) was kept: step 28,000
  (~28 epochs; validation BLEU 32.41, chrF2 63.99).

## Usage

The model expects the pretranslated input that the GLOSS-MT pipeline builds:

```bash
pip install git+https://github.com/GhanaOpenAI/gloss-mt.git
python -m spacy download en_core_web_sm
```

```python
from twi_translate.seq2seq import Seq2SeqTranslator

translator = Seq2SeqTranslator("ghanaopenai/gloss-mt-nllb-600m", num_beams=4)
result = translator.translate("The court announced the decision yesterday.")
print(result["translation"])
print(result["lookup_results"])  # dictionary phrases used
```

It also loads as a plain NLLB model (`AutoModelForSeq2SeqLM`, `src_lang="eng_Latn"`,
`forced_bos_token_id` of `twi_Latn`), but without the dictionary step quality is lower.

**Weights.** The default (`main`) weights are bf16, the precision the model was trained and is
served in. The original fp32 weights are on the `fp32` branch:
`AutoModelForSeq2SeqLM.from_pretrained("ghanaopenai/gloss-mt-nllb-600m", revision="fp32")`.

## The GLOSS-MT models

GLOSS-MT adds an editable dictionary to pretrained translation models: dictionary terms are
pretranslated in place in the source, and the model is fine-tuned once to use them.

| Model | Base | Fine-tune | chrF2 | Base chrF2 |
|---|---|---|:---:|:---:|
| [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b) | NLLB-200 3.3B | LoRA, merged | 50.80 | 47.74 |
| [`ghanaopenai/gloss-mt-nllb-1.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-1.3b) | NLLB-200 1.3B | full | 50.27 | 47.42 |
| [`ghanaopenai/gloss-mt-nllb-600m`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-600m) | NLLB-200 distilled 600M | full | 50.12 | 46.71 |
| [`ghanaopenai/gloss-mt-translategemma-4b`](https://huggingface.co/ghanaopenai/gloss-mt-translategemma-4b) | TranslateGemma 4B | LoRA adapter | 49.81 | no Twi zero-shot |
| [`ghanaopenai/gloss-mt-opus-mt-en-tw`](https://huggingface.co/ghanaopenai/gloss-mt-opus-mt-en-tw) | OPUS-MT en-tw | full | 48.00 | 42.52 |

All of them can be tried in the [GLOSS-MT Space](https://huggingface.co/spaces/ghanaopenai/gloss-mt).

## Scope

Trained on paragraph- and document-level prose (news, reports, education, narratives). Isolated
words or greetings without sentence context may translate poorly; give complete sentences.

## License

CC-BY-NC-4.0, as the base model and the training data.
