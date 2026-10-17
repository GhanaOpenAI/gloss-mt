# GLOSS-MT: Controllable Low-Resource Machine Translation with External Dictionaries

> **GLOSS-MT**: **G**rounded **L**exical **O**ffline **S**earch for **S**equence **T**ranslation

## Central Research Question

> **Can we add controllability to machine translation, and reduce the burden of fine-tuning, without degrading the performance of pretrained MT models?**

Every experiment in this work serves this question, which has three parts:
1. **Controllability.** Can an operator steer the translation (which term is used for a phrase) by editing an external dictionary, instead of retraining the model?
2. **Reduced fine-tuning burden.** Can a single, light fine-tune (LoRA on a few percent of the parameters, or a short full fine-tune of a small model) teach a model to use dictionary terms, so that later domain and terminology changes need dictionary entries rather than new parallel data and further training?
3. **No degradation.** Does the dictionary-grounded model translate at least as well as the pretrained MT model it starts from, measured on an independent document-level benchmark (nsanku-mt-benchmark) and on human-translated references?

To test whether the answer holds across model families and sizes, the same data and dictionary pipeline is applied to dedicated MT models (NLLB-200 600M, 1.3B and 3.3B; OPUS-MT en-tw) and a translation-specialised LLM (TranslateGemma 4B), and each is compared with its own pretrained base.

## 1. Preamble & Motivation

### 1.1 The Low-Resource Parallel Corpus Bottleneck
Building high-fidelity machine translation (MT) systems for low-resource languages—particularly African languages such as Twi (Akan)—remains fundamentally constrained by the scarcity of parallel sentence-level corpora. Supervised translation models depend heavily on large numbers of verified bilingual sentence pairs to learn semantic alignment, morphological rules, and lexical mappings simultaneously.

However, collecting sentence-level parallel text in low-resource environments is cost-prohibitive: it demands deep involvement from specialized bilingual native speakers, professional translators, and linguists who must manually translate, curate, and review long narrative documents. For many global languages with tens of millions of speakers, available parallel digital text constitutes only a fraction of what state-of-the-art MT systems require.

Pretrained multilingual models such as NLLB-200 already cover many of these languages, but adapting them to a new domain or to approved terminology conventionally means collecting more parallel data and fine-tuning again.

### 1.2 The Vocabulary Gap: Why Strong Models Do Not Reach Production
Several MT models that cover low-resource languages, NLLB-200 among them, perform strongly on benchmarks yet have seen little production use for these languages. A recurring reason is vocabulary: they often lack the correct words for the everyday entities of the settings they are deployed in, such as local institutions, government programmes, products, places, and newly coined terms. Low-resource languages are living, predominantly spoken languages, and their speakers continuously introduce new names for entities. Fine-tuning cannot keep a model current with this change: every new term would need new parallel data and another round of training, long after the term is in use. A deployable system needs a way to add or correct vocabulary as fast as the language changes, without retraining. GLOSS-MT provides it through an external dictionary that the model is trained once to follow.

### 1.3 Lowering Community Participation to Lexical Contributions
Our approach reduces the barrier to native-speaker participation. Instead of demanding thousands of annotated bilingual sentence pairs from scarce linguistic experts, our method reduces the requirement for language community participation to **contributing bilingual wordlists and terminology mappings**.

Lexical documentation (e.g., wordlists, bilingual glossaries, Webinary dictionaries, Wiktionary entries, and SIL language lexicons) has historically been the primary, most accessible mechanism through which linguistic field researchers and native language speakers document emerging and under-resourced languages. By separating the lexicon from the translation model:
1. Native speakers and community contributors need only provide translations for isolated noun phrases, idioms, and specialized terms.
2. The fine-tuned translation model handles grammatical morphology, subject-verb concord, tense marking, and discourse structure around the supplied terms, drawing on what it learned in pretraining.
This framework turns existing field-linguistic wordlists into usable machine translation assets.

### 1.4 Terminology Control and Hallucination Mitigation
Dedicated neural MT systems and general LLMs frequently mistranslate low-resource languages: they invent plausible-sounding but erroneous vocabulary, substitute loanwords indiscriminately, or omit nuanced technical terms. In high-stakes deployments—such as public health communication, statutory legal texts (e.g., the Universal Declaration of Human Rights), and national finance budgets—such lexical errors present real risks.

GLOSS-MT places verified terminology directly in the model's input. Each source chunk's noun phrases are looked up in an offline English–Twi dictionary, and every phrase found is replaced in place by its Twi translation (in angle brackets) before translation. The model is fine-tuned once to keep these terms and translate the sentence around them; phrases absent from the dictionary stay in English and are translated by the model as usual. Deploying organisations thus get **terminological control through the dictionary**: the term a phrase receives is the one the dictionary supplies. The control is learned, not enforced by constrained decoding, so it is measured directly (term adherence and dictionary-edit tests, Section 4 of the benchmark analysis): the fine-tuned models write 79–97% of the supplied terms, and 70–97% of the new terms after every entry is changed.

### 1.5 Zero-Parameter Test-Time Adaptation
In production environments, deploying institutions frequently require standardized or domain-specific terminology (e.g., ministry-approved budgetary phrasing, legal terms, medical terms, or dialectal variants). With conventional neural MT models, adapting to a new domain necessitates gathering parallel in-domain sentences and performing expensive fine-tuning.

GLOSS-MT enables **test-time terminology adaptation without parameter updates**:
- When deploying to a new domain, operators add or change the domain's terms in the external dictionary.
- The model incorporates the supplied vocabulary into its translation immediately, without a single gradient step.

The dictionary-edit (swap) test validates this mechanism: replacing every dictionary entry with a different term changes the output accordingly, with no retraining (Section 4 of the benchmark analysis). Measuring the quality gain from adding curated domain terminology for a new domain is left to future work.

### 1.6 Why Dedicated Translation Models
Dedicated translation models remain strong and inexpensive for low-resource languages: on the nsanku benchmark, the base NLLB-200 models (600M–3.3B) score within about 1 chrF2 of each other (46.71–47.74), so even the smallest is a strong starting point. They need a single encoder-decoder pass per chunk and run on modest hardware. GLOSS-MT therefore builds on these models (NLLB-200 at three sizes and OPUS-MT) and on a translation-specialised LLM (TranslateGemma 4B), fine-tuning each to translate partially pretranslated inputs. This keeps the community-contributed lexicon at the centre of the system while leaving inference as cheap as the base model's (Section 3 of the benchmark analysis).
