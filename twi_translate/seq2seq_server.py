"""HTTP API around the GLOSS-MT pipeline (a translation model + in-place dictionary terms), serving
several GLOSS-MT models; each request names the one to use.

Same endpoints and stream events as twi_translate.server, so the Space front end works with
either backend; there is no reasoning (the "reasoning" fields are always empty).

  GET  /models            the models offered: [{"id", "name", "params", "url", "loaded", "default"}]
  POST /translate         {"text": "...", "model": "nllb-3.3b"} -> {"translation", "reasoning": "", "lookups", "model"}
  POST /translate/stream  server-sent events: status(analyzing) -> lookups -> status(generating)
                          -> translation deltas, one per paragraph -> done
  GET  /health

The default model is loaded at start and stays loaded; the others load on first use, and at most
--max-loaded of them stay in GPU memory (the least recently used is unloaded first). All models
share one dictionary and one GPU lock.

  python -m twi_translate.seq2seq_server --port 8000
  python -m twi_translate.seq2seq_server --models nllb-3.3b,opus-mt --path opus-mt=checkpoints/opus-mt-en-tw-twi/final
"""

import argparse
import gc
import json
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

from .lookup import TwiLookup
from .seq2seq import GLOSSARY_DICT, ChunkedTranslator, chunk_text, glossary_for, protect_leaders

_PARAGRAPHS = re.compile(r"(\n\s*\n)")
NOT_FOUND = "Not found (translate from your own knowledge): "

# id -> name, size, Hugging Face repo (--path ID=DIR loads a local copy instead), page, backend.
# TranslateGemma's path is its LoRA adapter on google/translategemma-4b-it.
MODELS: Dict[str, Dict] = {
    "nllb-3.3b": {"name": "NLLB-200 3.3B", "params": "3.3B", "path": "ghanaopenai/gloss-mt-nllb-3.3b",
                  "url": "https://huggingface.co/ghanaopenai/gloss-mt-nllb-3.3b", "backend": "seq2seq"},
    "nllb-1.3b": {"name": "NLLB-200 1.3B", "params": "1.3B", "path": "ghanaopenai/gloss-mt-nllb-1.3b",
                  "url": "https://huggingface.co/ghanaopenai/gloss-mt-nllb-1.3b", "backend": "seq2seq"},
    "nllb-600m": {"name": "NLLB-200 600M", "params": "600M", "path": "ghanaopenai/gloss-mt-nllb-600m",
                  "url": "https://huggingface.co/ghanaopenai/gloss-mt-nllb-600m", "backend": "seq2seq"},
    "translategemma-4b": {"name": "TranslateGemma 4B", "params": "4B",
                          "path": "ghanaopenai/gloss-mt-translategemma-4b",
                          "url": "https://huggingface.co/ghanaopenai/gloss-mt-translategemma-4b",
                          "backend": "translategemma"},
    "opus-mt": {"name": "OPUS-MT en-tw", "params": "75M", "path": "ghanaopenai/gloss-mt-opus-mt-en-tw",
                "url": "https://huggingface.co/ghanaopenai/gloss-mt-opus-mt-en-tw", "backend": "seq2seq"},
}
DEFAULT_MODEL = "nllb-3.3b"
TRANSLATEGEMMA_BASE = "google/translategemma-4b-it"


def text_lookups(tr: ChunkedTranslator, text: str) -> List[Dict]:
    """Dictionary results for the text's phrases, in the {"phrases", "result"} shape of the
    tool-calling server ("- phrase -> twi" lines, then the phrases not in the dictionary)."""
    seen, found, missing = set(), [], []
    for seg, translate in protect_leaders(text):
        if not translate:
            continue
        for chunk in chunk_text(seg, tr.tok, tr.chunk_tokens)[0]:
            for r in glossary_for(chunk, tr.lookup):
                key = r["phrase"].lower()
                if key in seen:
                    continue
                seen.add(key)
                if r["found"]:
                    found.append(f"- {r['phrase']} -> {r['translations'][0]}")
                else:
                    missing.append(r["phrase"])
    if not seen:
        return []
    result = "\n".join(found + ([NOT_FOUND + ", ".join(missing)] if missing else []))
    return [{"phrases": sorted(seen), "result": result}]


class ModelPool:
    """The offered models, loaded on demand; callers hold `lock` while loading or translating."""

    def __init__(self, models: Dict[str, Dict], default: str, num_beams: int, max_loaded: int):
        self.models, self.default = models, default
        self.num_beams, self.max_loaded = num_beams, max(1, max_loaded)  # besides the default
        self.lookup = TwiLookup(GLOSSARY_DICT)
        self.loaded: "OrderedDict[str, ChunkedTranslator]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, model_id: Optional[str]) -> ChunkedTranslator:
        mid = model_id or self.default
        if mid not in self.models:
            raise KeyError(mid)
        if mid in self.loaded:
            self.loaded.move_to_end(mid)
            return self.loaded[mid]
        others = [k for k in self.loaded if k != self.default]  # least recently used first
        while mid != self.default and len(others) >= self.max_loaded:
            self.unload(others.pop(0))
        spec = self.models[mid]
        print(f"loading {mid} from {spec['path']}", flush=True)
        if spec["backend"] == "translategemma":
            from .translategemma import TranslateGemmaTranslator
            tr = TranslateGemmaTranslator(TRANSLATEGEMMA_BASE, adapter=spec["path"], lookup=self.lookup)
        else:
            from .seq2seq import Seq2SeqTranslator
            tr = Seq2SeqTranslator(spec["path"], lookup=self.lookup, num_beams=self.num_beams)
        self.loaded[mid] = tr
        return tr

    def unload(self, mid: str):
        print(f"unloading {mid}", flush=True)
        tr = self.loaded.pop(mid)
        del tr
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def describe(self) -> List[Dict]:
        return [{"id": k, "name": v["name"], "params": v["params"], "url": v["url"],
                 "loaded": k in self.loaded, "default": k == self.default} for k, v in self.models.items()]


def create_app(pool: ModelPool):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="GLOSS-MT")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    class Req(BaseModel):
        text: str
        model: Optional[str] = None
        temperature: float = 0.0  # accepted for compatibility; decoding is beam search or greedy

    def check(model_id: Optional[str]) -> str:
        mid = model_id or pool.default
        if mid not in pool.models:
            raise HTTPException(400, f"unknown model {mid!r}; available: {', '.join(pool.models)}")
        return mid

    @app.get("/health")
    def health():
        return {"status": "ok", "default": pool.default, "loaded": list(pool.loaded),
                "lookup_phrases": len(pool.lookup)}

    @app.get("/models")
    def models():
        return pool.describe()

    @app.post("/translate")
    def translate(req: Req):
        mid = check(req.model)
        with pool.lock:
            tr = pool.get(mid)
            lookups = text_lookups(tr, req.text)
            translation = tr.translate(req.text)["translation"]
        return {"translation": translation, "reasoning": "", "lookups": lookups, "model": mid}

    @app.post("/translate/stream")
    def translate_stream(req: Req):
        mid = check(req.model)

        def events():
            with pool.lock:
                yield {"type": "status", "stage": "analyzing"}
                if mid not in pool.loaded:
                    yield {"type": "status", "stage": "loading", "model": mid}
                tr = pool.get(mid)
                lookups = text_lookups(tr, req.text)
                yield {"type": "lookups", "lookups": lookups}
                yield {"type": "status", "stage": "generating"}
                out = ""
                for part in _PARAGRAPHS.split(req.text.strip()):
                    if _PARAGRAPHS.fullmatch(part):
                        piece = "\n\n"
                    elif part.strip():
                        piece = tr.translate(part)["translation"]
                    else:
                        continue
                    out += piece
                    yield {"type": "translation", "delta": piece}
                yield {"type": "done", "translation": out, "reasoning": "", "lookups": lookups, "model": mid}

        return StreamingResponse((f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events()),
                                 media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated ids to offer, in menu order")
    ap.add_argument("--default", default=DEFAULT_MODEL)
    ap.add_argument("--path", action="append", default=[], metavar="ID=PATH",
                    help="load a model from this path instead of its Hugging Face repo (repeatable)")
    ap.add_argument("--max-loaded", type=int, default=2, help="models besides the default kept in GPU memory")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    ids = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in ids + [args.default] if m not in MODELS]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; known: {', '.join(MODELS)}")
    models = {m: dict(MODELS[m]) for m in ids}
    for p in args.path:
        mid, _, path = p.partition("=")
        if mid not in models or not path:
            ap.error(f"--path {p!r}: expected ID=PATH with ID one of {', '.join(models)}")
        models[mid]["path"] = path
    if args.default not in models:
        ap.error(f"--default {args.default!r} is not in --models")

    import uvicorn

    pool = ModelPool(models, args.default, args.num_beams, args.max_loaded)
    pool.get(args.default)
    uvicorn.run(create_app(pool), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
