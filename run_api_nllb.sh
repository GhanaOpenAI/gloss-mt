#!/usr/bin/env bash
# GLOSS-MT API for the Space: the GLOSS-MT models (NLLB-200 3.3B by default; NLLB 1.3B and 600M,
# TranslateGemma 4B and OPUS-MT loaded on first use), same endpoints as the tool-calling API, on
# the port the public domain points to. Models not on Hugging Face load from their checkpoints.
PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
CK="$PROJ/checkpoints"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"
exec "$PROJ/.venv-seq2seq/bin/python" -m twi_translate.seq2seq_server --port "${PORT:-8000}" \
    --path nllb-1.3b=$CK/nllb-1.3b-twi/final \
    --path nllb-600m=$CK/nllb-600m-twi/final \
    --path translategemma-4b=$CK/translategemma-4b-twi/adapter \
    --path opus-mt=$CK/opus-mt-en-tw-twi/final
