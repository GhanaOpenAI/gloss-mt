"""Minimal HTTP API around TwiTranslator.

  POST /translate  {"text": "..."}  ->  {"translation": "...", "lookups": [...]}
  GET  /health
"""

import threading


def create_app(translator):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Twi translator (tool-calling)")
    lock = threading.Lock()  # in-process transformers backend is not re-entrant

    class Req(BaseModel):
        text: str
        temperature: float = 0.0

    @app.get("/health")
    def health():
        return {"status": "ok", "lookup_phrases": len(translator.lookup)}

    @app.post("/translate")
    def translate(req: Req):
        with lock:
            out = translator.translate(req.text, temperature=req.temperature)
        return {
            "translation": out["translation"],
            "lookups": [{"phrases": l["call"]["arguments"].get("phrases", []), "result": l["result"]}
                        for l in out["lookups"]],
        }

    return app


def serve(translator, host: str = "0.0.0.0", port: int = 8310):
    import uvicorn

    uvicorn.run(create_app(translator), host=host, port=port)
