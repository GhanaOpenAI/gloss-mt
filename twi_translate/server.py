"""Minimal HTTP API around TwiTranslator.

  POST /translate  {"text": "..."}  ->  {"translation": "...", "lookups": [...]}
  GET  /health
"""

import threading


def create_app(translator):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="Twi translator (tool-calling)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
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
            "reasoning": out.get("reasoning", ""),
            "lookups": [{"phrases": l["call"]["arguments"].get("phrases", []), "result": l["result"]}
                        for l in out["lookups"]],
        }

    @app.post("/translate/stream")
    def translate_stream_endpoint(req: Req):
        import json
        from fastapi.responses import StreamingResponse

        def event_generator():
            with lock:
                for event in translator.translate_stream(req.text, temperature=req.temperature):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def serve(translator, host: str = "0.0.0.0", port: int = 8310):
    import uvicorn

    uvicorn.run(create_app(translator), host=host, port=port)
