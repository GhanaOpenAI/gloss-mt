# GLOSS-MT: Grounding Generative LLMs with External Dictionaries for Low-Resource Machine Translation

> **GLOSS-MT**: **G**rounded **L**exical **O**ffline **S**earch for **S**equence **T**ranslation using LLMs

## Central Research Question

> **Can we add controllability to machine translation, and reduce the burden of fine-tuning, without degrading the performance of pretrained MT models?**

Every experiment in this work serves this question, which has three parts:
1. **Controllability.** Can an operator steer the translation (which term is used for a phrase) by editing an external dictionary, instead of retraining the model?
2. **Reduced fine-tuning burden.** Can a single, light fine-tune (LoRA on a few percent of the parameters, or a short full fine-tune of a small model) teach a model to use dictionary terms, so that later domain and terminology changes need dictionary entries rather than new parallel data and further training?
3. **No degradation.** Does the dictionary-grounded model translate at least as well as the pretrained MT model it starts from, measured on an independent document-level benchmark (nsanku-mt-benchmark)?

To test whether the answer holds across model families and sizes, the same data and dictionary pipeline is applied to dedicated MT models (NLLB-200 600M, 1.3B and 3.3B; OPUS-MT en-tw; MADLAD-400 3B), and a translation-specialised LLM (TranslateGemma 4B), and each is compared with its own pretrained base.

## 1. Preamble & Motivation

### 1.1 The Low-Resource Parallel Corpus Bottleneck
Building high-fidelity machine translation (MT) systems for low-resource languages—particularly African languages such as Twi (Akan)—remains fundamentally constrained by the scarcity of parallel sentence-level corpora. Traditional supervised sequence-to-sequence architectures and generative large language models (LLMs) depend heavily on millions of verified bilingual sentence pairs to learn semantic alignment, morphological rules, and lexical mappings simultaneously. 

However, collecting sentence-level parallel text in low-resource environments is cost-prohibitive: it demands deep involvement from specialized bilingual native speakers, professional translators, and linguists who must manually translate, curate, and review long narrative documents. For many global languages with tens of millions of speakers, available parallel digital text constitutes only a fraction of what state-of-the-art MT systems require.

### 1.2 Lowering Community Participation to Lexical Contributions
Our approach introduces a structural paradigm shift that radically reduces the barrier to native-speaker participation. Instead of demanding thousands of annotated bilingual sentence pairs from scarce linguistic experts, our method reduces the requirement for language community participation to **contributing bilingual wordlists and terminology mappings**.

Lexical documentation (e.g., wordlists, bilingual glossaries, Webinary dictionaries, Wiktionary entries, and SIL language lexicons) has historically been the primary, most accessible mechanism through which linguistic field researchers and native language speakers document emerging and under-resourced languages. By decoupling lexical knowledge retrieval from syntactic sentence generation:
1. Native speakers and community contributors need only provide translations for isolated noun phrases, idioms, and specialized terms.
2. The fine-tuned LLM leverages its pre-trained general linguistic reasoning to handle grammatical morphology, subject-verb concord, tense marking, and discourse structure around the retrieved terms.
This framework turns hundreds of years of static field-linguistic wordlists into actionable, dynamic machine translation assets.

### 1.3 Deterministic Controllability and Hallucination Mitigation
Even state-of-the-art closed and open LLMs (as well as dedicated neural MT systems) frequently hallucinate when translating low-resource languages. They invent plausible-sounding but erroneous vocabulary, substitute loanwords indiscriminately, or omit nuanced technical terms. In mission-critical deployments—such as public health communication, statutory legal texts (e.g., the Universal Declaration of Human Rights), and national finance budgets—such lexical hallucinations present unacceptable real-world risks.

By formulating translation as an agentic tool-calling trajectory (`input text` → `extract noun phrases` → `tool query` → `grounded lexical context` → `linguistic reasoning` → `final translation`), our model is constrained to adhere to verified reference vocabulary. The model learns to incorporate user-supplied and verified terminology where it fits, and only fall back to parametric knowledge when a term is explicitly marked as absent. This provides deploying organizations with **deterministic terminological control**.

### 1.4 Zero-Parameter Test-Time Domain Adaptation
In production environments, deploying institutions frequently require standardized or domain-specific terminology (e.g., ministry-approved budgetary phrasing, legal jurisprudence, medical terms, or dialectal variants). With conventional neural MT models, adapting to a new domain necessitates gathering parallel in-domain sentences and performing expensive fine-tuning.

Our architecture enables **zero-shot, zero-parameter test-time domain adaptation**:
- When deploying to an unseen target domain, operators simply extract the target domain's noun phrases and augment the external lookup dictionary.
- The model immediately incorporates the newly supplied vocabulary into its reasoning and translation output without requiring a single gradient step or parameter update.

We empirically validate this capability by demonstrating that augmenting the lookup dictionary with target-domain terminology yields immediate, measurable improvements across benchmark evaluations (Citizens' Budget, Universal Declaration of Human Rights, Watchtower prose, and traditional Akan Folk-Tales).

### 1.5 Extending Dictionary Grounding Beyond Tool-Calling LLMs
Tool calling requires an instruction-tuned LLM. Dedicated translation models, which remain strong and cheap for low-resource languages, take only source text. We therefore also study a **tool-free** variant of GLOSS-MT: the same dictionary is applied before translation by replacing the source's dictionary phrases with their target-language translations in place, and a translation model (NLLB-200 at three sizes, OPUS-MT, MADLAD-400 and TranslateGemma) is fine-tuned to translate these partially pretranslated inputs. This keeps the community-contributed lexicon at the centre of the system while reducing inference to a single encoder-decoder pass (Section 3 of the benchmark analysis).
