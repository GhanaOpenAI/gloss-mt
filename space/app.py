import gc
import os
import re
import sys
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twi_translate.seq2seq import Seq2SeqTranslator, pretranslate, glossary_for

MODELS = {
    "GLOSS-MT NLLB-200 3.3B (Default)": {
        "repo_id": "ghanaopenai/gloss-mt-nllb-3.3b",
        "default_glossary": True,
        "description": "NLLB-200 3.3B fine-tuned with in-place offline dictionary grounding.",
    },
    "NLLB-200 3.3B (Base Model)": {
        "repo_id": "facebook/nllb-200-3.3B",
        "default_glossary": False,
        "description": "Base Meta NLLB-200 3.3B model.",
    },
    "NLLB-200 1.3B (Base Model)": {
        "repo_id": "facebook/nllb-200-1.3B",
        "default_glossary": False,
        "description": "Base Meta NLLB-200 1.3B model (lighter & faster).",
    },
    "NLLB-200 600M (Distilled)": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "default_glossary": False,
        "description": "Distilled 600M model for fastest inference.",
    },
}

DEFAULT_MODEL_KEY = "GLOSS-MT NLLB-200 3.3B (Default)"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

current_model_key = None
current_translator = None


def get_translator(model_key: str, use_glossary: bool):
    global current_model_key, current_translator
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL_KEY
    config = MODELS[model_key]
    repo_id = config["repo_id"]

    if (
        current_translator is not None
        and current_model_key == model_key
        and current_translator.glossary == use_glossary
    ):
        return current_translator

    if current_translator is not None and current_model_key != model_key:
        print(f"Unloading {current_model_key} to free GPU memory...")
        del current_translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Loading {repo_id} (glossary={use_glossary}) on {device}...")
    current_translator = Seq2SeqTranslator(model=repo_id, device=device, glossary=use_glossary)
    current_model_key = model_key
    return current_translator


# Preload default model at startup
print("Preloading default model...")
get_translator(DEFAULT_MODEL_KEY, use_glossary=True)
print("Default model ready!")

EXAMPLES = [
    [
        "## Community Health Notice\n\n"
        "The district health directorate will hold a free screening exercise next week.\n\n"
        "- **Date:** Monday, 12 May\n"
        "- **Venue:** Kumasi Central Market\n"
        "- **Services:** blood pressure checks, malaria testing and eye care\n\n"
        "All residents are encouraged to attend.",
        DEFAULT_MODEL_KEY,
        True,
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
        True,
    ],
    [
        "# How to Register Your Farm\n\n"
        "1. Visit the nearest district agriculture office.\n"
        "2. Bring your **Ghana Card** and a photograph of your farm.\n"
        "3. Fill in the registration form and submit it to the officer.\n\n"
        "*Registered farmers will receive subsidised seeds and fertiliser.*",
        DEFAULT_MODEL_KEY,
        True,
    ],
    [
        "## Article 1\n\n"
        "> All human beings are born free and equal in dignity and rights.\n\n"
        "They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.",
        DEFAULT_MODEL_KEY,
        True,
    ],
]


def translate(text: str, model_choice: str, use_glossary: bool):
    if not text or not text.strip():
        return "", "", ""

    translator = get_translator(model_choice, use_glossary=use_glossary)
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
        lookups_table = "*No glossary terms used or matched.*" if use_glossary else "*Glossary grounding disabled.*"

    if use_glossary and translator.lookup:
        try:
            gloss = glossary_for(text, translator.lookup)
            pretrans_text = pretranslate(text, gloss)
        except Exception:
            pretrans_text = text
    else:
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
            "State-of-the-art English to Asante Twi translation powered by **GLOSS-MT** models "
            "with in-place offline glossary grounding.\n\n"
            "**Supported by:** [Ghana NLP](https://ghananlp.org) & [Ghana Open AI](https://huggingface.co/ghanaopenai) | "
            "**Model:** [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b) | "
            "**Dataset:** [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi) | "
            "**Code:** [`GitHub`](https://github.com/GhanaOpenAI/gloss-mt)"
        )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL_KEY,
                label="Translation Model",
                scale=3,
            )
            glossary_toggle = gr.Checkbox(
                value=True,
                label="Use Glossary Grounding",
                info="Pre-translates domain terms before generation",
                scale=1,
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
            inputs=[input_text, model_selector, glossary_toggle],
            outputs=[output_text, lookups_display, pretrans_display],
            fn=translate,
            cache_examples=False,
            label="Sample Document Inputs (Click to test)",
        )

        translate_btn.click(
            fn=translate,
            inputs=[input_text, model_selector, glossary_toggle],
            outputs=[output_text, lookups_display, pretrans_display],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
