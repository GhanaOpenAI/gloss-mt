"""HTTP API around the GLOSS-MT seq2seq pipeline (NLLB + in-place dictionary terms).

Same endpoints and stream events as twi_translate.server, so the Space front end works with
either backend; there is no reasoning (the "reasoning" fields are always empty).

  POST /translate         {"text": "..."} -> {"translation", "reasoning": "", "lookups"}
  POST /translate/stream  server-sent events: status(analyzing) -> lookups -> status(generating)
                          -> translation deltas, one per paragraph -> done
  GET  /health

  python -m twi_translate.seq2seq_server --model ghanaopenai/gloss-mt-nllb-3.3b --port 8000
"""

import argparse
import json
import re
import threading
from typing import Dict, List

from .seq2seq import Seq2SeqTranslator, chunk_text, glossary_for, protect_leaders

_PARAGRAPHS = re.compile(r"(\n\s*\n)")
NOT_FOUND = "Not found (translate from your own knowledge): "


def text_lookups(tr: Seq2SeqTranslator, text: str) -> List[Dict]:
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


def create_app(tr: Seq2SeqTranslator):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="GLOSS-MT (NLLB-200 3.3B + Glossary)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    lock = threading.Lock()  # one GPU model, one request at a time

    class Req(BaseModel):
        text: str
        temperature: float = 0.0  # accepted for compatibility; decoding is beam search

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "gloss-mt-nllb-3.3b", "lookup_phrases": len(tr.lookup)}

    @app.post("/translate")
    def translate(req: Req):
        with lock:
            lookups = text_lookups(tr, req.text)
            translation = tr.translate(req.text)["translation"]
        return {"translation": translation, "reasoning": "", "lookups": lookups}

    @app.post("/translate/stream")
    def translate_stream(req: Req):
        def events():
            with lock:
                yield {"type": "status", "stage": "analyzing"}
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
                yield {"type": "done", "translation": out, "reasoning": "", "lookups": lookups}

        return StreamingResponse((f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events()),
                                 media_type="text/event-stream")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ghanaopenai/gloss-mt-nllb-3.3b")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    tr = Seq2SeqTranslator(args.model, num_beams=args.num_beams)
    uvicorn.run(create_app(tr), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
