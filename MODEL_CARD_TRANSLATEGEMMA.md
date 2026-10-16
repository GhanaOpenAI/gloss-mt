---
language:
- en
- tw
license: other
library_name: peft
tags:
- translation
- Akan
- Asante Twi
- translategemma
- gemma3
- lora
- peft
- glossary
- terminology
- GLOSS-MT
base_model: google/translategemma-4b-it
datasets:
- ghanaopenai/gloss-mt-parallel-en-twi
metrics:
- chrf
- bleu
pipeline_tag: translation
---

# GLOSS-MT (TranslateGemma 4B + Glossary)

**`ghanaopenai/gloss-mt-translategemma-4b`** is an English → Asante Twi (Akan) translation model: a LoRA adapter for [`google/translategemma-4b-it`](https://huggingface.co/google/translategemma-4b-it) to
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
TranslateGemma 4B + this LoRA adapter  en → ak (fine-tuned)  ──► Twi chunk with the terms in brackets
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
| GLOSS-MT NLLB-200 600M | 50.12 | 12.23 | 0.99 | 49.81 | 36.61 | 62.80 | 51.25 |
| Khaya AI | 49.91 | 12.79 | 1.04 | 47.74 | 36.22 | 65.78 | 49.92 |
| **GLOSS-MT TranslateGemma 4B** | **49.81** | **12.24** | **1.06** | **50.03** | **35.48** | **62.54** | **51.19** |
| GLOSS-MT OPUS-MT en-tw | 48.00 | 10.91 | 0.97 | 46.13 | 35.15 | 60.62 | 50.09 |

TranslateGemma has no Twi language code and, zero-shot, does not produce Twi; with this adapter (target code `ak`) and the dictionary it reaches 49.81 chrF2, its output running slightly long (length ratio 1.06).

## Controllability

On 300 human-translated Asante Twi Bible passages (2,766 dictionary terms), the model writes 96.6% of the dictionary terms in its output (26.2% of the same terms appear without the dictionary). When every entry is replaced by a different term, 97.3% of the new terms appear and 1.9% of the old ones remain. With the dictionary it scores chrF2 37.15 on these passages, against 32.13 without it.

## Training

- **Data:** 31,909 line-aligned chunks (≤ 192 tokens) of English–Twi documents. Each English noun
  phrase was aligned to its span in the gold Twi with Gemini, and the aligned phrases are bracketed
  on both sides (`english_pretranslated` → `twi_bracketed`), so the model learns to keep the
  bracketed terms.
- **Method:** LoRA r=32, α=64 (60M trainable parameters), bf16, AdamW, lr 2e-4, batch 8 × 4
  gradient accumulation. The dictionary terms ride inside TranslateGemma's own translation prompt
  (source `en`, target `ak`).
- **Checkpoint choice:** every 1,000 steps the model translated 200 fixed validation chunks; the
  checkpoint with the best mean of BLEU, chrF2 and 100·e^(−loss) was kept: step 3,000
  (~3 epochs; validation BLEU 33.73, chrF2 64.96). Decoding is greedy.

## Usage

The model expects the pretranslated input that the GLOSS-MT pipeline builds:

```bash
pip install git+https://github.com/GhanaOpenAI/gloss-mt.git
python -m spacy download en_core_web_sm
```

```python
from twi_translate.translategemma import TranslateGemmaTranslator

translator = TranslateGemmaTranslator("google/translategemma-4b-it",
                                      adapter="ghanaopenai/gloss-mt-translategemma-4b")
result = translator.translate("The court announced the decision yesterday.")
print(result["translation"])
print(result["lookup_results"])  # dictionary phrases used
```

The adapter loads with `peft.PeftModel.from_pretrained` on top of `google/translategemma-4b-it`
(`AutoModelForImageTextToText`), with TranslateGemma's chat template (`source_lang_code="en"`,
`target_lang_code="ak"`).

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

The adapter is a derivative of TranslateGemma and is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms); the training data is CC-BY-NC-4.0, so use is non-commercial.
