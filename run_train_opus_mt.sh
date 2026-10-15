#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="Helsinki-NLP/opus-mt-en-tw"
OUT="$PROJ/checkpoints/opus-mt-en-tw-twi"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== Full fine-tune of Helsinki-NLP/opus-mt-en-tw (MarianMT) on ghanaopenai/gloss-mt-parallel-en-twi (chunks split to 512 tokens, best by composite of BLEU/chrF2/loss; best + last checkpoints kept) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

$VENV/bin/python training/train_opus_mt.py \
    --model "$MODEL" \
    --out "$OUT" \
    --data $PROJ/repo/build/glossary-chunks-v2/dataset \
    --max-src 512 \
    --max-tgt 512 \
    --epochs 30 \
    --early-stopping-patience 5 \
    --batch-size 32 \
    --grad-accum 1 \
    --lr 1e-4 \
    --eval-steps 1000 \
    >> "$LOG_DIR/train-opus-mt-en-tw.log" 2>&1

echo "=== OPUS-MT en-tw fine-tuning complete! ==="
