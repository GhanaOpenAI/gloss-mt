"""Tool specification and prompt templates for Twi translation tool-calling.

Everything the model sees (tool schema, system prompt, the format of tool results fed
back to it) is defined here, and is shared by the dataset builder (dataset_builder/build_dataset.py)
and the deployment runtime (twi_translate/runtime.py). Training and inference therefore see identical
text, which is what makes the fine-tuned model usable as-is.
"""

from typing import Any, Dict, List

TOOL_NAME = "lookup_twi_translation"

# OpenAI / Hugging Face chat-template compatible tool specification.
LOOKUP_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Look up Twi (Akan) translations of English noun phrases in an offline English-Twi "
            "dictionary (Gemini translations from AfriSpeech/shola-machine-translations). "
            "Pass the key noun phrases of the text you need to translate, without leading articles "
            "or determiners. Returns the candidate Twi translations for each phrase; a phrase with "
            "an empty list was not found and must be translated from your own knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phrases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "English noun phrases, lowercase, articles removed "
                    "(e.g. ['temperature difference', 'cylinder', 'road construction']).",
                },
            },
            "required": ["phrases"],
        },
    },
}

TOOLS: List[Dict[str, Any]] = [LOOKUP_TOOL]

SYSTEM_PROMPT = (
    "You are an expert English-to-Twi (Akan) translator. First call the lookup_twi_translation "
    "tool with the key English noun phrases of the text to get reference Twi terms from the "
    "dictionary. Then reason step by step about the grammar, idioms and word choice, using the "
    "dictionary results where they fit the context and your own knowledge where a phrase was not "
    "found or the dictionary term does not fit. Finally give only the Twi translation, keeping "
    "the original Markdown formatting."
)

# Same wording as the user turns in ghananlpcommunity/twi-english-reasoning-translation.
USER_TEMPLATE = (
    "Translate the following English text into Twi (Akan). Return only the translation with no "
    "additional text.\n\n{text}"
)
USER_TEMPLATES: List[str] = [USER_TEMPLATE]


def format_tool_result(results: List[Dict[str, Any]]) -> str:
    """Render lookup results into the tool message that is fed back to the LLM.

    `results` is the output of TwiLookup.lookup_many: a list of
    {"phrase": str, "translations": [str, ...]} dicts, where an empty list means the
    phrase was not found in the dataset.
    """
    found = [r for r in results if r["translations"]]
    missing = [r["phrase"] for r in results if not r["translations"]]
    lines = [f"Twi lookup results ({len(found)} of {len(results)} phrases found):"]
    for r in found:
        lines.append(f"- {r['phrase']} -> {' | '.join(r['translations'])}")
    if not found:
        lines.append("- (no translations found)")
    if missing:
        lines.append("Not found (translate from your own knowledge): " + ", ".join(missing))
    return "\n".join(lines)
