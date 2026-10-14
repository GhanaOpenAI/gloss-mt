"""TranslateGemma backend for the chunked glossary pipeline (twi_translate.seq2seq).

TranslateGemma (google/translategemma-*-it, Gemma 3) only accepts its own translation prompt:
one user turn whose content carries source/target language codes and the text. The
dictionary terms therefore ride inside that text, pretranslated in place in the chunk
(translate_input), which a LoRA fine-tune (training/train_translategemma.py) teaches the model to use. Twi has no code
in the template, so the target is Akan ("ak"). glossary=False on the base model gives the
zero-shot TranslateGemma baseline.
"""

from typing import Dict, List, Optional

from .lookup import TwiLookup
from .seq2seq import ChunkedTranslator, translate_input

SOURCE_LANG, TARGET_LANG = "en", "ak"
END_OF_TURN = "<end_of_turn>"


def prompt_messages(chunk: str, results: Optional[List[Dict]]) -> List[Dict]:
    return [{"role": "user", "content": [{
        "type": "text", "source_lang_code": SOURCE_LANG, "target_lang_code": TARGET_LANG,
        "text": translate_input(chunk, results, prefix=""),
    }]}]


def render_prompt(tok, chunk: str, results: Optional[List[Dict]]) -> str:
    return tok.apply_chat_template(prompt_messages(chunk, results), tokenize=False, add_generation_prompt=True)


def load_translategemma(path: str, **kwargs):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(path, **kwargs)


class TranslateGemmaTranslator(ChunkedTranslator):
    def __init__(self, model: str, adapter: Optional[str] = None, lookup: Optional[TwiLookup] = None,
                 device: Optional[str] = None, chunk_tokens: int = 192, max_new_tokens: int = 1024,
                 batch_size: int = 16, glossary: bool = True):
        import torch
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        tok.padding_side = "left"
        super().__init__(tok, lookup, chunk_tokens, glossary)
        self.torch = torch
        self.device = device or "cuda:0"
        self.model = load_translategemma(model, dtype=torch.bfloat16, device_map={"": self.device})
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.eos = [tok.eos_token_id, tok.convert_tokens_to_ids(END_OF_TURN)]

    def encode(self, chunk: str, results: Optional[List[Dict]]) -> str:
        return render_prompt(self.tok, chunk, results)

    def generate(self, inputs: List[str]) -> List[str]:
        order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
        out = [""] * len(inputs)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([inputs[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t.strip()
        return out
