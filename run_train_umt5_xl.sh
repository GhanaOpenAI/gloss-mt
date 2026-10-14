#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="google/umt5-xl"
OUT="$PROJ/checkpoints/umt5-xl-twi"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== [1/2] Full fine-tune of google/umt5-xl (chunked glossary-conditioned translation, Adafactor, bf16) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

$VENV/bin/python training/train_seq2seq.py \
    --model "$MODEL" \
    --data data \
    --out "$OUT" \
    --chunk-tokens 192 \
    --max-src 1024 \
    --epochs 5 \
    --early-stopping-patience 3 \
    --batch-size 8 \
    --grad-accum 4 \
    --lr 5e-4 \
    --eval-steps 500 \
    >> "$LOG_DIR/train-umt5-xl.log" 2>&1

echo "=== [2/2] Evaluating on validation and test ==="
for split in validation test; do
    $VENV/bin/python training/eval_seq2seq.py --model "$OUT/final" --data data/$split.jsonl \
        >> "$LOG_DIR/eval-umt5-xl.log" 2>&1
done
echo "=== umT5-XL fine-tuning and evaluation complete! ==="
