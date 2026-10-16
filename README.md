# GLOSS-MT: English → Twi translation with an editable dictionary

GLOSS-MT adds an external, editable dictionary to pretrained translation models. Before a text is
translated, its English noun phrases are looked up in an offline English → Twi dictionary, and
every phrase found is replaced **in place** by its Twi translation. The model, fine-tuned once to
use these terms, translates the rest of the sentence around them. Terminology can then be changed
by editing the dictionary, without new parallel data or retraining.

The same recipe is applied to dedicated translation models (NLLB-200 at three sizes, OPUS-MT,
MADLAD-400) and a translation-specialised LLM (TranslateGemma 4B), and each is compared with its
own pretrained base.

| Artifact | Where |
|---|---|
| Models | [`gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b), [`gloss-mt-nllb-1.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-1.3b), [`gloss-mt-nllb-600m`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-600m), [`gloss-mt-translategemma-4b`](https://huggingface.co/ghanaopenai/gloss-mt-translategemma-4b), [`gloss-mt-opus-mt-en-tw`](https://huggingface.co/ghanaopenai/gloss-mt-opus-mt-en-tw) |
| Training data | [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi) |
| Live demo (Space) | [`ghanaopenai/gloss-mt`](https://huggingface.co/spaces/ghanaopenai/gloss-mt): switch between the models |
| Dictionary | `twi_translate/data/twi_dictionary_single.tsv.gz`, 613,259 English → Twi phrases, one translation each, offline |
| Benchmark | [nsanku-mt-benchmark](https://github.com/GhanaNLP/nsanku-mt-benchmark) ([leaderboard](https://huggingface.co/spaces/ghananlpcommunity/nsanku-mt-benchmark)) |
| Paper notes | [`paper/PREAMBLE.md`](paper/PREAMBLE.md), [`paper/BENCHMARK_ANALYSIS.md`](paper/BENCHMARK_ANALYSIS.md) |

## How it works

```
English text
  │  split into line-aligned chunks of at most 192 tokens (128 for OPUS-MT)
  ▼
spaCy noun phrases ──► offline dictionary lookup
  │
  ▼
found phrases replaced by their Twi translation in angle brackets, in place
  "Our people have courage, but courage without foundation is foolishness."
  "Our <nnipa> have <akokoduro>, but <akokoduro> without <fapem> is <nkwasiasɛm>."
  │
  ▼
fine-tuned translation model ──► Twi chunk with the terms in brackets
  │  brackets removed, chunks re-joined
  ▼
Twi text
```

Phrases not in the dictionary stay plain English for the model to translate. Markdown structure
(headings, lists, tables) is kept line by line, and table-of-contents dot leaders are kept out of
the model and restored unchanged.

> **Input scope:** GLOSS-MT is trained on **paragraph- and document-level text**. Isolated words or
> greetings (e.g. a standalone *"Welcome"*) without sentence context can translate poorly; give
> complete sentences or paragraphs.

## Results (nsanku-mt-benchmark, English → Asante Twi)

Document-level chrF2 over 84 documents in four domains (each domain weighs the same in the mean):

| Model | chrF2 | BLEU | Base model's chrF2 |
|---|:---:|:---:|:---:|
| Gemini 3.8 Flash | 54.64 | 17.25 | |
| Google Translate | 51.26 | 15.67 | |
| **GLOSS-MT NLLB-200 3.3B** | **50.80** | 13.11 | 47.74 |
| **GLOSS-MT NLLB-200 1.3B** | **50.27** | 12.82 | 47.42 |
| **GLOSS-MT NLLB-200 600M** | **50.12** | 12.23 | 46.71 |
| Khaya AI | 49.91 | 12.79 | |
| **GLOSS-MT TranslateGemma 4B** | **49.81** | 12.24 | no Twi output zero-shot |
| **GLOSS-MT OPUS-MT en-tw** | **48.00** | 10.91 | 42.52 |

Every fine-tune scores above its pretrained base on chrF2 (+2.85 to +5.48); on BLEU the NLLB
fine-tunes are 0.5–1.0 below their bases. **Controllability** is measured directly
(`training/eval_controllability.py`): on human-translated Asante Twi Bible passages the fine-tuned
models write 79–97% of the dictionary terms, and when every dictionary entry is replaced by a
different term, 70–97% of the new terms appear, with no retraining. Details in
[`paper/BENCHMARK_ANALYSIS.md`](paper/BENCHMARK_ANALYSIS.md).

## Quick start

```bash
pip install git+https://github.com/GhanaOpenAI/gloss-mt.git
pip install torch transformers sentencepiece spacy
python -m spacy download en_core_web_sm
```

```python
from twi_translate.seq2seq import Seq2SeqTranslator

tr = Seq2SeqTranslator("ghanaopenai/gloss-mt-nllb-3.3b", num_beams=4)   # or -1.3b, -600m, -opus-mt-en-tw
out = tr.translate("The district assembly approved the new budget for schools and clinics.")
out["translation"], out["lookup_results"]   # Twi text, dictionary phrases used
```

TranslateGemma loads its adapter on the base model:

```python
from twi_translate.translategemma import TranslateGemmaTranslator

tr = TranslateGemmaTranslator("google/translategemma-4b-it", adapter="ghanaopenai/gloss-mt-translategemma-4b")
```

`glossary=False` on either translator skips the dictionary (the model's own translation), and a
custom dictionary can be passed as `lookup=TwiLookup(path)`.

### HTTP API

The API behind the Space serves all GLOSS-MT models, loading each from Hugging Face on first use:

```bash
pip install fastapi uvicorn pydantic
python -m twi_translate.seq2seq_server --port 8000
```

| Endpoint | |
|---|---|
| `GET /models` | the models offered |
| `POST /translate` | `{"text": "...", "model": "nllb-1.3b"}` → `{"translation", "lookups", "model"}` |
| `POST /translate/stream` | the same as server-sent events: lookups, then one translation delta per paragraph |

## Data

`scripts/build_chunked_dataset.py` builds the training data from English–Twi documents with gold
human Twi translations (15,134 training documents):

1. Each document is split into line-aligned English/Twi lines, packed into chunks of at most 192
   tokens (31,909 training chunks).
2. spaCy extracts each chunk's English noun phrases; Gemini aligns each phrase to its span in the
   gold Twi chunk (kept only if the span occurs verbatim, on the same line).
3. The aligned pairs are bracketed on both sides: `english_pretranslated` (the training input,
   with the Twi spans in place) and `twi_bracketed` (the target).

The dictionary (`twi_dictionary_single.tsv.gz`) has exactly one translation per phrase:
`scripts/rebuild_dictionary_single.py` re-translated the entries that had several candidates, and
`scripts/sync_glossary_with_gold.py` adds every aligned phrase–span pair of the dataset.

## Training

| Model | Script | Fine-tune |
|---|---|---|
| NLLB-200 3.3B / 1.3B / 600M | `training/train_seq2seq.py` (`run_train_nllb_*.sh`) | LoRA r=32 (3.3B, merged with `training/merge_seq2seq_lora.py`) or full (1.3B, 600M) |
| OPUS-MT en-tw | `training/train_opus_mt.py` (`run_train_opus_mt.sh`) | full |
| TranslateGemma 4B | `training/train_translategemma.py` (`run_train_translategemma_4b.sh`) | LoRA r=32 |
| MADLAD-400 3B | `training/train_madlad.py` (`run_train_madlad_3b.sh`) | LoRA r=32 |

Every 1,000 steps the model translates 200 fixed validation chunks
(`training/gen_eval.py`), and the checkpoint with the best mean of BLEU, chrF2 and
100·e^(−loss) is kept, with early stopping.

## Evaluation

- `training/eval_seq2seq.py`: BLEU / chrF2 of a model through the full pipeline.
- `training/eval_controllability.py`: term adherence with the dictionary, without it, and with
  every entry swapped for a different term; `--jsonl` runs it on a human-translated set such as
  the Asante Twi Bible passages of `scripts/build_bible_testset.py` (the Bible text is not
  redistributed; `results/controllability/` keeps its metrics only).
- Document-level benchmark: [nsanku-mt-benchmark](https://github.com/GhanaNLP/nsanku-mt-benchmark).

## Layout

| Path | Purpose |
|---|---|
| `twi_translate/` | `seq2seq.py` (chunking, lookup, pretranslation, NLLB/OPUS-MT/MADLAD translator), `translategemma.py`, `lookup.py` (dictionary), `seq2seq_server.py` (HTTP API), `data/` |
| `scripts/` | dataset, dictionary and test-set builders |
| `training/` | fine-tuning, checkpoint selection and evaluation |
| `paper/` | research question, method and results |
| `results/` | controllability results |
| `space/` | the Hugging Face Space front end |
| `tests/` | offline tests (`python -m pytest tests -q`) |

## Licence

CC-BY-NC-4.0, inherited from the data sources. Non-commercial use only. The TranslateGemma adapter
is also subject to the Gemma Terms of Use.
