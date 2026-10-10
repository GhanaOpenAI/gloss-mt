# Twi translation with tool calling + reasoning

Teach an LLM to translate English into Twi (Akan) like a careful translator:

1. **look up** the key English noun phrases with a `lookup_twi_translation` tool call,
2. **read** the dictionary results (phrases that aren't found come back empty),
3. **reason** about grammar, idioms and word choice, using the dictionary terms where they fit,
4. **translate**.

This repo contains all of it: the code that builds the dataset, the offline dictionary the tool
uses, the fine-tuning and merging scripts, an evaluation script, and the **deployment runtime**
that runs a fine-tuned model end to end. The runtime parses the tool call, does the lookup, feeds
the results back and returns the translation.

| Artifact | Where |
|---|---|
| Dataset | [`ghanaopenai/twi-translation-tool-calling`](https://huggingface.co/datasets/ghanaopenai/twi-translation-tool-calling) (private) |
| Base model | `google/gemma-4-31B-it` (LoRA, language layers only) |
| Dictionary | `twi_translate/data/twi_dictionary.tsv.gz`, 478,822 English→Twi phrases, offline |

## How it works at inference

```
user text ──► model ──► <|tool_call>call:lookup_twi_translation{phrases:[...]}<tool_call|>
                              │
                    twi_translate.TwiLookup  (local dictionary, no network)
                              │
          tool result ──► model ──► <|channel>thought ...reasoning...<channel|> Twi translation
```

The model only writes text. `twi_translate.TwiTranslator` runs the loop:

1. render the prompt (Gemma 4 template, `enable_thinking=True`, exactly as in training),
2. parse the tool call from the model output,
3. look the phrases up in the bundled dictionary,
4. append the result as a `tool` message (same template as the training data),
5. generate again to get the reasoning and translation.

## Quick start (deployment)

```bash
pip install -e ".[local]"          # in-process with transformers (GPU)
twi-translate --model <merged-model-dir-or-hf-id> -v "The temperature difference between the inner and outer surface of the cylinder."
```

Or serve with vLLM and use the client backend:

```bash
pip install -e ".[client]"
vllm serve <merged-model> --served-model-name twi-translator --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
twi-translate --base-url http://localhost:8000/v1 --model twi-translator -v "Did you hear the matter in court?"
```

As an HTTP service (`POST /translate {"text": ...}`):

```bash
pip install -e ".[local,server]"
twi-translate --model <merged-model> --serve --port 8310
```

From Python:

```python
from twi_translate import TwiTranslator, TransformersBackend   # or OpenAIBackend(base_url, model)
tr = TwiTranslator(TransformersBackend("<merged-model>"))
out = tr.translate("Did you hear the matter in court?")
out["translation"], out["reasoning"], out["lookups"]
```

The runtime understands both Gemma 4 (`<|tool_call>call:...`) and Hermes/Qwen
(`<tool_call>{json}</tool_call>`) tool-call formats.

## The dataset

Built by `dataset_builder/build_dataset.py` from the English→Twi rows (`pristine`,
`ghana_chat_en2tw`) of
[`ghananlpcommunity/twi-english-reasoning-translation`](https://huggingface.co/datasets/ghananlpcommunity/twi-english-reasoning-translation).
In those rows the Twi is a gold human reference, and Gemini wrote a numbered reasoning trace
explaining it.

| Part of each example | Source |
|---|---|
| user turn | the original English text, Markdown included |
| tool call | English phrases **quoted in the reasoning** (e.g. `'national progress' is 'ɔman nkɔso'`) that occur in the source text and are noun phrases (spaCy), with articles and possessives removed, at most 4 words |
| tool result | those phrases looked up in the offline dictionary; not-found phrases are listed as such |
| final turn | `reasoning_content` = the original reasoning, `content` = the gold Twi translation |

The trajectory is `system → user → assistant(tool_calls) → tool → assistant(reasoning + translation)`,
so the dictionary results come before, and feed, the linguistic reasoning.

| split | examples | with tool call | phrases | found |
|---|---|---|---|---|
| train | 15,134 | 14,431 | 139,747 | 41.2% |
| validation | 332 | 322 | 3,166 | 40.6% |
| test | 538 | 513 | 4,864 | 40.8% |

The dictionary was exported from the Gemini translations in
[`AfriSpeech/shola-machine-translations`](https://huggingface.co/datasets/AfriSpeech/shola-machine-translations)
(`gemini-3.6-flash-one`, then up to 3 options from `gemini-3.6-flash`). The same file builds the
training data and serves the tool at deployment, so the model sees identical tool output in both.
Rebuild it with `python -m twi_translate.lookup --rebuild`.

## Training

```bash
# Unsloth (bf16 LoRA r=32 on the language model; vision/audio towers frozen)
python training/train_sft.py --backend unsloth --model google/gemma-4-31B-it --data data \
    --out checkpoints/gemma4-31b-twi-tool
# merge the adapter into the bf16 base -> standalone model for vLLM / transformers
python training/merge_lora.py --base google/gemma-4-31B-it \
    --adapter checkpoints/gemma4-31b-twi-tool/adapter --out checkpoints/gemma4-31b-twi-tool/merged
```

Loss is only on the tokens the model generates at inference: the tool call (including the
`<|tool_response>` stop token), then reasoning, translation and `<turn|>`. Conversations are rendered with
`enable_thinking=True`, the same as the runtime. Examples run 2.9k tokens at the median and 5.4k at most
(`--max-len 6144`). `--backend hf` uses plain transformers + PEFT instead of Unsloth.

## Evaluation

`training/evaluate.py` runs held-out test examples through the real runtime (vLLM backend) and
reports tool-call rate, phrase precision/recall against the reference phrases, chrF++ and BLEU.
`--no-tools` gives a direct-translation baseline.

## Layout

| Path | Purpose |
|---|---|
| `twi_translate/` | installable runtime: `schema.py` (tool + prompts), `lookup.py` (dictionary), `runtime.py` (loop, parsers, backends), `cli.py`, `server.py`, `data/` |
| `dataset_builder/` | `phrases.py` (phrases from reasoning), `build_dataset.py` |
| `training/` | `train_sft.py`, `merge_lora.py`, `evaluate.py` |
| `tests/` | offline tests (`python -m pytest tests -q`) |

## Licence

CC-BY-NC-4.0, inherited from the data sources. Non-commercial use only.
