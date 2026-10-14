---
title: GLOSS-MT
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: English to Twi translator, NLLB-200 3.3B + dictionary
---

# GLOSS-MT (Akan / Asante Twi)

Google Translate-style interface for English-to-Twi translation powered by NLLB-200 3.3B fine-tuned with an offline English–Twi dictionary: key phrases are looked up and pretranslated in place, then the model translates the text (served on an NVIDIA H200).

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)  
**Supported by:** [Ghana NLP](https://ghananlp.org)  
**Model:** [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b) (the tool-calling Gemma 4 E4B version is [`ghanaopenai/gloss-mt`](https://huggingface.co/ghanaopenai/gloss-mt))  
**Code:** [`GhanaOpenAI/gloss-mt`](https://github.com/GhanaOpenAI/gloss-mt)

### 💡 Best Practice & Input Scope
GLOSS-MT is optimized for **long-form, paragraph-level, and document texts** (news articles, reports, educational prose, and narratives). Translating isolated single-word fragments or elliptical greetings (e.g., standalone *"Welcome"*) without full subject/predicate context may yield degraded results. For optimal translation quality, provide complete grammatical sentences or cohesive paragraphs.
