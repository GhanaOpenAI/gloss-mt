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
base_model: facebook/nllb-200-3.3B
datasets:
- ghanaopenai/twi-translation-glossary-chunks
metrics:
- chrf
- bleu
pipeline_tag: translation
---

# GLOSS-MT (NLLB-200 3.3B + Glossary)

**GLOSS-MT NLLB** (`ghanaopenai/gloss-mt-nllb-3.3b`) is an English → Asante Twi (Akan) translation
model: [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) fine-tuned with LoRA
(merged into the weights) to translate text in which dictionary terms have already been
**pretranslated in place**. It is the tool-free member of the GLOSS-MT family: the same offline
dictionary as [GLOSS-MT (Gemma 4 E4B)](https://huggingface.co/ghanaopenai/gloss-mt), applied by a
fixed pipeline instead of tool calls, so there is no reasoning step and no LLM server.

- **Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)
- **Supported by:** [Ghana NLP](https://ghananlp.org)
- **Code:** [GhanaOpenAI/gloss-mt](https://github.com/GhanaOpenAI/gloss-mt) (`twi_translate.seq2seq`)
- **Live demo:** [Hugging Face Spaces: GLOSS-MT](https://huggingface.co/spaces/ghanaopenai/gloss-mt)
- **Training data:** [`ghanaopenai/twi-translation-glossary-chunks`](https://huggingface.co/datasets/ghanaopenai/twi-translation-glossary-chunks)
- **Dictionary:** 545,633 English → Twi phrases, one translation each (`twi_dictionary_single.tsv.gz`)

## How it works

```
English text
  │  split into line-aligned chunks of at most 192 tokens
  ▼
spaCy noun phrases ──► offline dictionary lookup
  │
  ▼
found phrases replaced by their Twi translation, in place
  "Our people have courage, but courage without foundation is foolishness."
  "Our nnipa have akokoduro, but akokoduro without fapem is nkwasiasɛm."
  │
  ▼
NLLB-200 3.3B (fine-tuned)  eng_Latn → twi_Latn  ──► Twi chunk, chunks re-joined
```

Phrases not in the dictionary stay in English. The model learned to translate the rest of the
sentence around the pretranslated terms. Dot leaders of tables of contents are kept out of the
model and restored unchanged.

## Benchmark (nsanku-mt-benchmark, English → Asante Twi)

Document-level chrF2 on the [nsanku MT benchmark](https://github.com/GhanaNLP/nsanku-mt-benchmark)
(84 documents, 4 domains; a language score is the mean of its per-dataset scores):

| Model | chrF2 | BLEU | Length ratio | Finance (3) | Folktales (75) | Watchtower (5) | UDHR (1) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemini 3.8 Flash | 54.64 | 17.25 | 1.02 | 52.50 | 40.33 | 69.28 | 56.44 |
| Google Translate | 51.26 | 15.67 | 1.01 | 48.06 | 37.54 | 67.15 | 52.30 |
| Khaya AI | 49.91 | 12.79 | 1.04 | 47.74 | 36.22 | 65.78 | 49.92 |
| **GLOSS-MT (NLLB-200 3.3B + Glossary)** | **49.78** | **13.93** | **0.99** | 46.97 | 36.16 | 65.55 | 50.46 |
| GLOSS-MT (Gemma 4 E4B + Tool Calling) | 49.75 | 11.45 | 1.19 | 51.23 | 36.23 | 60.75 | 50.80 |
| NLLB-200 3.3B (base) | 47.74 | 14.15 | 1.01 | 45.25 | 36.76 | 56.89 | 52.05 |

Fine-tuning plus the dictionary adds 2.0 chrF2 over the base NLLB-200 3.3B (+8.7 on Watchtower),
and matches the tool-calling GLOSS-MT Gemma 4 E4B with one encoder-decoder pass per chunk.

## Training

- **Data:** the 15,134 training documents of `ghanaopenai/twi-translation-tool-calling`, cut into
  32,806 line-aligned chunks (≤ 192 tokens), each with its dictionary phrases pretranslated in place.
- **Method:** LoRA r=32, α=64 on every attention and feed-forward projection (69M trainable
  parameters, 2%), plus the embedding of an added `"\n"` token that keeps markdown line structure;
  bf16, AdamW, lr 2e-4, effective batch 32.
- **Checkpoint choice:** every 1,000 steps the model translated 200 fixed validation chunks; the
  checkpoint with the best mean of BLEU, chrF2 and 100·e^(−loss) was kept: step 8,000
  (~8 epochs; validation BLEU 46.82, chrF2 68.55, loss 0.80).

## Usage

The model expects the pretranslated input that the GLOSS-MT pipeline builds:

```bash
pip install git+https://github.com/GhanaOpenAI/gloss-mt.git
python -m spacy download en_core_web_sm
```

```python
from twi_translate.seq2seq import Seq2SeqTranslator

translator = Seq2SeqTranslator("ghanaopenai/gloss-mt-nllb-3.3b", num_beams=4)
result = translator.translate("The court announced the decision yesterday.")
print(result["translation"])
print(result["lookup_results"])  # dictionary phrases used
```

It also loads as a plain NLLB model (`AutoModelForSeq2SeqLM`, `src_lang="eng_Latn"`,
`forced_bos_token_id` of `twi_Latn`), but without the dictionary step quality is lower.

## Scope

Trained on paragraph- and document-level prose (news, reports, education, narratives). Isolated
words or greetings without sentence context may translate poorly; give complete sentences.

## License

CC-BY-NC-4.0, as the base model and the training data.
