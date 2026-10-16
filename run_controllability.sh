#!/usr/bin/env bash
# Controllability test (training/eval_controllability.py) of the GLOSS-MT fine-tunes and two base
# models given the same dictionary; results in repo/results/controllability/.
PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
VENV="$PROJ/.venv-seq2seq"
CK="$PROJ/checkpoints"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ/repo"
run() { echo "[$(date "+%F %T")] $1"; $VENV/bin/python training/eval_controllability.py --name "$@" \
    --data $PROJ/repo/build/glossary-chunks-v2/dataset --n 300 >> "$PROJ/logs/controllability.log" 2>&1; echo "rc=$?"; }
run gloss-mt-opus-mt-en-tw --model $CK/opus-mt-en-tw-twi/final
run opus-mt-en-tw-base --model Helsinki-NLP/opus-mt-en-tw
run gloss-mt-nllb-3.3b --model $CK/nllb-3.3b-twi/final --batch-size 8
run nllb-200-3.3b-base --model facebook/nllb-200-3.3B --batch-size 8
run gloss-mt-translategemma-4b --backend translategemma --model google/translategemma-4b-it \
    --adapter $CK/translategemma-4b-twi/adapter --batch-size 4
echo "[$(date "+%F %T")] done"
