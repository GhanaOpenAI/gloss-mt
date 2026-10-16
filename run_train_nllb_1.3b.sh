#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="facebook/nllb-200-1.3B"
OUT="$PROJ/checkpoints/nllb-1.3b-twi"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== Full fine-tune (Adafactor) of facebook/nllb-200-1.3B on ghanaopenai/gloss-mt-parallel-en-twi (best by composite of BLEU/chrF2/loss; best + last checkpoints kept) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

$VENV/bin/python training/train_seq2seq.py \
    --model "$MODEL" \
    --out "$OUT" \
    --data $PROJ/repo/build/glossary-chunks-v2/dataset \
    --max-src 1024 \
    --epochs 30 \
    --early-stopping-patience 5 \
    --lora-r 0 \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 1e-4 \
    --eval-steps 1000 \
    >> "$LOG_DIR/train-nllb-1.3b.log" 2>&1

echo "=== facebook/nllb-200-1.3B fine-tuning complete! ==="
