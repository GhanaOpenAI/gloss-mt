#!/usr/bin/env bash
# GLOSS-MT API (NLLB-200 3.3B + Glossary) for the Space: same endpoints as the tool-calling API,
# on the port the public domain points to. Replaces `twi_translate.cli --serve` (Gemma via vLLM).
PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"
exec "$PROJ/.venv-seq2seq/bin/python" -m twi_translate.seq2seq_server \
    --model ghanaopenai/gloss-mt-nllb-3.3b --port "${PORT:-8000}"
