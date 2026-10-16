#!/usr/bin/env bash
# Retry of the NLLB 3.3B validation runs, which ran out of GPU memory at batch 8 next to the
# NLLB 600M training: after the Bible base run, with batch 2.
P=/mnt/volume_d2wey28/projects/twi-translation-tool-calling
CK=$P/checkpoints
V="--data $P/repo/build/glossary-chunks-v2/dataset --n 300"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $P/repo
run() { echo "[$(date "+%F %T")] $1"; $P/.venv-seq2seq/bin/python training/eval_controllability.py --name "$@" >> $P/logs/controllability.log 2>&1; echo "rc=$?"; }
until grep -q "all-done" $P/logs/controllability-rest.out; do sleep 60; done
run gloss-mt-nllb-3.3b --model $CK/nllb-3.3b-twi/final --batch-size 2 $V
run nllb-200-3.3b-base --model facebook/nllb-200-3.3B --batch-size 2 $V
echo "[$(date "+%F %T")] all-done"
