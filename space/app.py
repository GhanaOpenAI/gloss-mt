import os
import re
import sys
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twi_translate.seq2seq import Seq2SeqTranslator, pretranslate, glossary_for

MODEL_ID = os.environ.get("MODEL_ID", "ghanaopenai/gloss-mt-nllb-3.3b")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_ID} on {device} ({dtype})...")
translator = Seq2SeqTranslator(model=MODEL_ID, device=device)
print("GLOSS-MT Translator initialized successfully!")

EXAMPLES = [
    [
        "## Community Health Notice\n\n"
        "The district health directorate will hold a free screening exercise next week.\n\n"
        "- **Date:** Monday, 12 May\n"
        "- **Venue:** Kumasi Central Market\n"
        "- **Services:** blood pressure checks, malaria testing and eye care\n\n"
        "All residents are encouraged to attend."
    ],
    [
        "### 2024 Budget Allocations\n\n"
        "| Sector | Amount (GH₵ million) | Main focus |\n"
        "|---|---|---|\n"
        "| Education | 1,250 | Free senior high school |\n"
        "| Health | 980 | Rural clinics and nurses |\n"
        "| Agriculture | 640 | Fertiliser for farmers |\n\n"
        "The government says these sectors will receive priority this year."
    ],
    [
        "# How to Register Your Farm\n\n"
        "1. Visit the nearest district agriculture office.\n"
        "2. Bring your **Ghana Card** and a photograph of your farm.\n"
        "3. Fill in the registration form and submit it to the officer.\n\n"
        "*Registered farmers will receive subsidised seeds and fertiliser.*"
    ],
    [
        "## Article 1\n\n"
        "> All human beings are born free and equal in dignity and rights.\n\n"
        "They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood."
    ],
]


def translate(text: str):
    if not text or not text.strip():
        return "", "", ""

    res = translator.translate(text)
    translation = res.get("translation", "")
    lookup_results = res.get("lookup_results", [])

    # Format lookups into a markdown table
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

    # Show pretranslated text
    try:
        gloss = glossary_for(text, translator.lookup)
        pretrans_text = pretranslate(text, gloss)
    except Exception:
        pretrans_text = text

    return translation, lookups_table, pretrans_text


custom_css = """
.container { max-width: 1000px; margin: auto; padding-top: 10px; }
.output-box textarea { font-size: 16px !important; line-height: 1.6 !important; }
"""

with gr.Blocks(title="GLOSS-MT: English → Twi Translator", css=custom_css) as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown(
            "# 🌐 GLOSS-MT: English → Twi (Akan)\n"
            "State-of-the-art English to Asante Twi translation powered by **GLOSS-MT NLLB-200 3.3B** "
            "with in-place offline glossary grounding.\n\n"
            "**Supported by:** [Ghana NLP](https://ghananlp.org) & [Ghana Open AI](https://huggingface.co/ghanaopenai) | "
            "**Model:** [`ghanaopenai/gloss-mt-nllb-3.3b`](https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b) | "
            "**Dataset:** [`ghanaopenai/gloss-mt-parallel-en-twi`](https://huggingface.co/datasets/ghanaopenai/gloss-mt-parallel-en-twi) | "
            "**Code:** [`GitHub`](https://github.com/GhanaOpenAI/gloss-mt)"
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
                    gr.Markdown("### ⚙️ Pre-translated Model Input (`<terms>` inserted in-place)")
                    pretrans_display = gr.Textbox(label="Model Input", lines=6, interactive=False)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text],
            outputs=[output_text, lookups_display, pretrans_display],
            fn=translate,
            cache_examples=False,
            label="Sample Document Inputs (Click to test)",
        )

        translate_btn.click(
            fn=translate,
            inputs=[input_text],
            outputs=[output_text, lookups_display, pretrans_display],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
