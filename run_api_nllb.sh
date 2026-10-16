#!/usr/bin/env bash
# GLOSS-MT API for the Space: the GLOSS-MT models from Hugging Face (NLLB-200 3.3B by default;
# NLLB 1.3B and 600M, TranslateGemma 4B and OPUS-MT loaded on first use), on the port the public
# domain points to.
PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"
exec "$PROJ/.venv-seq2seq/bin/python" -m twi_translate.seq2seq_server --port "${PORT:-8000}"
