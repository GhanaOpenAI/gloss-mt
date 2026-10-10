---
language:
- en
- ak
license: cc-by-nc-4.0
task_categories:
- translation
- text-generation
tags:
- twi
- akan
- ghana
- tool-calling
- function-calling
- reasoning
- chain-of-thought
- machine-translation
pretty_name: Twi Translation Tool-Calling with Reasoning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Twi Translation Tool-Calling with Reasoning

SFT data that teaches an LLM to translate English into Twi (Akan) like a careful translator:

1. **look up** the key English noun phrases in a dictionary (`lookup_twi_translation` tool call),
2. **read the results** (a tool message; phrases that aren't in the dictionary come back empty),
3. **reason** about grammar, idioms and word choice, using the dictionary terms where they fit,
4. **translate**.

Each row is a full agent trajectory in the standard chat format with `tools`:

```
system -> user -> assistant(tool_calls) -> tool(results) -> assistant(reasoning_content + content)
```

## Columns

For browsing, each row has the trajectory as flat columns, in order:

| column | content |
|---|---|
| `english` | source text |
| `tool_phrases` | phrases the model looks up (the tool-call arguments) |
| `tool_result` | what the tool returns (fed back to the model) |
| `reasoning` | step-by-step linguistic reasoning, written after the lookup |
| `translation` | gold Twi translation |

`messages` + `tools` hold the same content in the chat format used for training. The reasoning is
in the final assistant message's `reasoning_content` field, separate from `content`.

## Where each part comes from

| Part | Source |
|---|---|
| English text, reasoning, Twi translation | English→Twi rows (`pristine`, `ghana_chat_en2tw`) of [ghananlpcommunity/twi-english-reasoning-translation](https://huggingface.co/datasets/ghananlpcommunity/twi-english-reasoning-translation). The Twi side is the gold human reference, and the reasoning explains it |
| Tool-call phrases | the English phrases the reasoning quotes (e.g. `'national progress' is 'ɔman nkɔso'`), kept only if they occur in the source text and are noun phrases (spaCy), with articles and possessives removed, at most 4 words |
| Tool results | those phrases looked up in an offline English→Twi dictionary of 478,822 phrases, built from the Gemini translations in [AfriSpeech/shola-machine-translations](https://huggingface.co/datasets/AfriSpeech/shola-machine-translations) (`gemini-3.6-flash-one`, falling back to up to 3 options from `gemini-3.6-flash`) |

Because the tool results come from the same dictionary the deployed tool uses, a model trained on
this data receives exactly the kind of tool output it saw in training. Dictionary entries are
machine candidates, not verified translations, so the reasoning and gold translations
sometimes use a different term. That teaches the model to treat lookups as hints, not ground truth.

## Splits

| split | examples | with tool call | phrases looked up | found in dictionary |
|---|---|---|---|---|
| train | 15,134 | 14,431 | 139,747 | 41.2% |
| validation | 332 | 322 | 3,166 | 40.6% |
| test | 538 | 513 | 4,864 | 40.8% |

Splits are a stable hash of the source row `id`. About 70% of rows carry Markdown formatting
(headings, lists, tables, bold and italic) that the translation preserves.

## Example (abridged)

```json
{"messages": [
  {"role": "system", "content": "You are an expert English-to-Twi (Akan) translator. First call the lookup_twi_translation tool ..."},
  {"role": "user", "content": "Translate the following English text into Twi (Akan). Return only the translation with no additional text.\n\n..."},
  {"role": "assistant", "content": "", "tool_calls": [{"id": "call_000123", "type": "function",
    "function": {"name": "lookup_twi_translation", "arguments": {"phrases": ["cocoa heaps", "proper fermentation", "inspection"]}}}]},
  {"role": "tool", "tool_call_id": "call_000123", "name": "lookup_twi_translation",
   "content": "Twi lookup results (2 of 3 phrases found):\n- cocoa heaps -> koko sumie | koko ekuo | koko boaboa ano\n- inspection -> nhwehwɛmu\nNot found (translate from your own knowledge): proper fermentation"},
  {"role": "assistant", "reasoning_content": "1. 'Sɛnea nhwehwɛmu kyerɛ no' translates 'According to research' ...", "content": "Sɛnea nhwehwɛmu kyerɛ no, ..."}
 ],
 "tools": [{"type": "function", "function": {"name": "lookup_twi_translation", "parameters": {"type": "object", "properties": {"phrases": {"type": "array", "items": {"type": "string"}}}, "required": ["phrases"]}}}],
 "id": "...", "source": "pristine", "markdown": "heading,bullet_list", "num_phrases": 3, "num_found": 2}
```

`reasoning_content` is the standard field that chat templates (e.g. Gemma 4, Qwen 3) render as the
model's thinking. With Gemma 4, render with `enable_thinking=True` so the thought channel opens
after the tool result.

## Code

The dataset builder, the offline dictionary, the training script and a ready-to-run
deployment runtime (tool-call parsing → dictionary lookup → result fed back → translation, with
transformers, vLLM or an HTTP API) are in the accompanying GitHub repository.

## Licence

CC-BY-NC-4.0, inherited from both sources. Non-commercial use only.
