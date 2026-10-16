---
title: GLOSS-MT
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: English to Twi translator, GLOSS-MT models + dictionary
---

# GLOSS-MT (Akan / Asante Twi)

Google Translate-style interface for English-to-Twi translation with the GLOSS-MT models, translation models fine-tuned with an offline English–Twi dictionary: key phrases are looked up and pretranslated in place, then the model translates the text (served on an NVIDIA H200). A menu switches between the models: NLLB-200 3.3B (default), NLLB-200 1.3B, NLLB-200 600M, TranslateGemma 4B and OPUS-MT en-tw.

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)  
**Supported by:** [Ghana NLP](https://ghananlp.org)  
**Default model:** [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b)  
**Code:** [`GhanaOpenAI/gloss-mt`](https://github.com/GhanaOpenAI/gloss-mt)

### 💡 Best Practice & Input Scope
GLOSS-MT is optimized for **long-form, paragraph-level, and document texts** (news articles, reports, educational prose, and narratives). Translating isolated single-word fragments or elliptical greetings (e.g., standalone *"Welcome"*) without full subject/predicate context may yield degraded results. For optimal translation quality, provide complete grammatical sentences or cohesive paragraphs.
