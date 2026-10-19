---
title: GLOSS-MT
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
short_description: English to Twi translator (GLOSS-MT NLLB-3.3B)
---

# GLOSS-MT (English → Akan / Asante Twi)

Google Translate-style interface for English-to-Twi translation powered by **GLOSS-MT NLLB-200 3.3B**, fine-tuned with in-place offline dictionary grounding.

- **Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)
- **Supported by:** [Ghana NLP](https://ghananlp.org)
- **Model:** [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b)
- **Dataset:** [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi)
- **Code:** [`GhanaOpenAI/gloss-mt`](https://github.com/GhanaOpenAI/gloss-mt)

### 💡 Input Scope
GLOSS-MT is optimized for **long-form, paragraph-level, and document texts** (news articles, reports, educational prose, and narratives). Translating isolated single-word fragments or elliptical greetings without full context may yield degraded results. Provide complete grammatical sentences or cohesive paragraphs for best results.
