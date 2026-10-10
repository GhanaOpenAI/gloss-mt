"""Command line: translate English to Twi with a fine-tuned tool-calling model.

  # in-process (transformers)
  twi-translate --model ghananlpcommunity/<model> "Did you hear the matter in court?"

  # against a running vLLM / OpenAI-compatible server
  twi-translate --base-url http://localhost:8300/v1 --model twi-translator "..."

  # HTTP service
  twi-translate --model <model> --serve --port 8310
"""

import argparse
import os
import sys


def build_translator(args):
    from .lookup import TwiLookup
    from .runtime import OpenAIBackend, TransformersBackend, TwiTranslator

    if args.base_url:
        backend = OpenAIBackend(args.base_url, args.model, args.api_key)
    else:
        backend = TransformersBackend(args.model, max_new_tokens=args.max_new_tokens)
    return TwiTranslator(backend, TwiLookup())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="English text (reads stdin if omitted)")
    ap.add_argument("--model", required=True, help="model path / HF id, or served model name with --base-url")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; omit to run in-process")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("-v", "--verbose", action="store_true", help="print tool calls and results")
    ap.add_argument("--serve", action="store_true", help="run an HTTP API instead of translating once")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8310)
    args = ap.parse_args()

    translator = build_translator(args)
    if args.serve:
        from .server import serve
        serve(translator, args.host, args.port)
        return
    text = args.text if args.text else sys.stdin.read()
    out = translator.translate(text, verbose=args.verbose)
    if args.verbose and out.get("reasoning"):
        print(f"[reasoning]\n{out['reasoning']}\n")
    print(out["translation"])


if __name__ == "__main__":
    main()
