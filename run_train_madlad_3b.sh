#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="google/madlad400-3b-mt"
OUT="$PROJ/checkpoints/madlad-3b-twi"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== LoRA fine-tune (r=32) of google/madlad400-3b-mt on ghanaopenai/gloss-mt-parallel-en-twi (best by composite of BLEU/chrF2/loss; best + last checkpoints kept) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

$VENV/bin/python training/train_madlad.py \
    --model "$MODEL" \
    --out "$OUT" \
    --data $PROJ/repo/build/glossary-chunks-v2/dataset \
    --max-src 1024 \
    --epochs 30 \
    --early-stopping-patience 5 \
    --lora-r 32 \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 2e-4 \
    --eval-steps 1000 \
    >> "$LOG_DIR/train-madlad-3b.log" 2>&1

echo "=== google/madlad400-3b-mt fine-tuning complete! ==="
