import gc
import os
import re
import sys
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twi_translate.lookup import TwiLookup
from twi_translate.seq2seq import (
    Seq2SeqTranslator,
    GLOSSARY_DICT,
    pretranslate,
    glossary_for,
)

MODELS = {
    "GLOSS-MT NLLB-200 3.3B (Default)": {
        "repo_id": "ghanaopenai/gloss-mt-nllb-3.3b",
        "backend": "seq2seq",
        "params": "3.3B",
        "description": "NLLB-200 3.3B fine-tuned with in-place offline glossary grounding.",
    },
    "GLOSS-MT NLLB-200 1.3B": {
        "repo_id": "ghanaopenai/gloss-mt-nllb-1.3b",
        "backend": "seq2seq",
        "params": "1.3B",
        "description": "Mid-sized 1.3B model balancing high accuracy and fast generation.",
    },
    "GLOSS-MT NLLB-200 600M": {
        "repo_id": "ghanaopenai/gloss-mt-nllb-600m",
        "backend": "seq2seq",
        "params": "600M",
        "description": "Distilled 600M model optimized for ultra-fast, lightweight inference.",
    },
    "GLOSS-MT OPUS-MT en-tw": {
        "repo_id": "ghanaopenai/gloss-mt-opus-mt-en-tw",
        "backend": "seq2seq",
        "params": "75M",
        "description": "75M Marian architecture fine-tuned with dictionary grounding.",
    },
    "GLOSS-MT TranslateGemma 4B": {
        "repo_id": "ghanaopenai/gloss-mt-translategemma-4b",
        "base_model": "google/translategemma-4b-it",
        "backend": "translategemma",
        "params": "4B",
        "description": "Google TranslateGemma-4B fine-tuned with in-place glossary LoRA.",
    },
}

DEFAULT_MODEL_KEY = "GLOSS-MT NLLB-200 3.3B (Default)"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading shared offline Twi dictionary...")
shared_lookup = TwiLookup(GLOSSARY_DICT)

current_model_key = None
current_translator = None


def get_translator(model_key: str):
    global current_model_key, current_translator
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL_KEY
    spec = MODELS[model_key]

    if current_translator is not None and current_model_key == model_key:
        return current_translator

    if current_translator is not None:
        print(f"Unloading {current_model_key} to free GPU memory...")
        del current_translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Loading {model_key} ({spec['repo_id']}) on {device}...")
    if spec["backend"] == "translategemma":
        from twi_translate.translategemma import TranslateGemmaTranslator
        current_translator = TranslateGemmaTranslator(
            spec["base_model"],
            adapter=spec["repo_id"],
            lookup=shared_lookup,
            device=device,
        )
    else:
        current_translator = Seq2SeqTranslator(
            model=spec["repo_id"],
            lookup=shared_lookup,
            device=device,
        )

    current_model_key = model_key
    return current_translator


# Preload default model
print(f"Preloading default model: {DEFAULT_MODEL_KEY}...")
get_translator(DEFAULT_MODEL_KEY)
print("GLOSS-MT Suite ready!")

EXAMPLES = [
    [
        "## Community Health Notice\n\n"
        "The district health directorate will hold a free screening exercise next week.\n\n"
        "- **Date:** Monday, 12 May\n"
        "- **Venue:** Kumasi Central Market\n"
        "- **Services:** blood pressure checks, malaria testing and eye care\n\n"
        "All residents are encouraged to attend.",
        DEFAULT_MODEL_KEY,
    ],
    [
        "### 2024 Budget Allocations\n\n"
        "| Sector | Amount (GH₵ million) | Main focus |\n"
        "|---|---|---|\n"
        "| Education | 1,250 | Free senior high school |\n"
        "| Health | 980 | Rural clinics and nurses |\n"
        "| Agriculture | 640 | Fertiliser for farmers |\n\n"
        "The government says these sectors will receive priority this year.",
        DEFAULT_MODEL_KEY,
    ],
    [
        "# How to Register Your Farm\n\n"
        "1. Visit the nearest district agriculture office.\n"
        "2. Bring your **Ghana Card** and a photograph of your farm.\n"
        "3. Fill in the registration form and submit it to the officer.\n\n"
        "*Registered farmers will receive subsidised seeds and fertiliser.*",
        DEFAULT_MODEL_KEY,
    ],
    [
        "## Article 1\n\n"
        "> All human beings are born free and equal in dignity and rights.\n\n"
        "They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.",
        DEFAULT_MODEL_KEY,
    ],
]


def translate(text: str, model_choice: str):
    if not text or not text.strip():
        return "", "", ""

    translator = get_translator(model_choice)
    res = translator.translate(text)
    translation = res.get("translation", "")
    lookup_results = res.get("lookup_results", [])

    matched_rows = []
    seen = set()
    for r in lookup_results:
        ph = r.get("phrase", "").strip()
        trans = r.get("translations", [])
        found = r.get("found", False)
        if found and trans and ph.lower() not in seen:
            seen.add(ph.lower())
            matched_rows.append(f"| **{ph}** | `{trans[0]}` |")

    if matched_rows:
        lookups_table = "| English Term | Twi Translation |\n|---|---|\n" + "\n".join(matched_rows)
    else:
        lookups_table = "*No glossary terms matched from the offline dictionary.*"

    try:
        gloss = glossary_for(text, translator.lookup)
        pretrans_text = pretranslate(text, gloss)
    except Exception:
        pretrans_text = text

    return translation, lookups_table, pretrans_text


custom_css = """
.container { max-width: 1050px; margin: auto; padding-top: 10px; }
.output-box textarea { font-size: 16px !important; line-height: 1.6 !important; }
"""

with gr.Blocks(title="GLOSS-MT: English → Twi Translator", css=custom_css) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            "# 🌐 GLOSS-MT: English → Twi (Akan)\n"
            "State-of-the-art English to Asante Twi translation powered by the **GLOSS-MT** model suite "
            "with in-place offline glossary grounding.\n\n"
            "**Supported by:** [Ghana NLP](https://ghananlp.org) & [Ghana Open AI](https://huggingface.co/ghanaopenai) | "
            "**Models:** [`ghanaopenai`](https://huggingface.co/models?search=ghanaopenai%2Fgloss-mt) | "
            "**Dataset:** [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi) | "
            "**Code:** [`GitHub`](https://github.com/GhanaOpenAI/gloss-mt)"
        )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL_KEY,
                label="Select GLOSS-MT Model",
                info="Switch between the fine-tuned GLOSS-MT models",
                scale=3,
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="English Input",
                    placeholder="Enter English text to translate (supports Markdown, tables, lists, etc.)...",
                    lines=8,
                )
                translate_btn = gr.Button("Translate to Twi", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Asante Twi Translation",
                    lines=8,
                    show_copy_button=True,
                    elem_classes=["output-box"],
                )

        with gr.Accordion("🔍 Inspected Terms & In-Place Grounding", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📖 Matched Glossary Terms")
                    lookups_display = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Model Input (`<terms>` inserted in-place)")
                    pretrans_display = gr.Textbox(label="Pre-translated Source", lines=6, interactive=False)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text, model_selector],
            outputs=[output_text, lookups_display, pretrans_display],
            fn=translate,
            cache_examples=False,
            label="Sample Document Inputs (Click to test)",
        )

        translate_btn.click(
            fn=translate,
            inputs=[input_text, model_selector],
            outputs=[output_text, lookups_display, pretrans_display],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
