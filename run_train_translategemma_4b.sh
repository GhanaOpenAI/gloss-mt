#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="google/translategemma-4b-it"
OUT="$PROJ/checkpoints/translategemma-4b-twi"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== [1/2] LoRA fine-tune of google/translategemma-4b-it (r=32, glossary-conditioned, best by composite of BLEU/chrF2/loss; best + last checkpoints kept) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
# GEMINI_API_KEY: GeminiScore is reported in the final eval only (not used to pick checkpoints)
[ -f "$PROJ/.env" ] && set -a && . "$PROJ/.env" && set +a
cd "$PROJ/repo"

$VENV/bin/python training/train_translategemma.py \
    --model "$MODEL" \
    --data data \
    --out "$OUT" \
    --chunk-tokens 192 \
    --epochs 30 \
    --early-stopping-patience 5 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-r 32 \
    --eval-steps 500 \
    >> "$LOG_DIR/train-translategemma-4b.log" 2>&1

echo "=== [2/2] Evaluating on validation and test ==="
for split in validation test; do
    $VENV/bin/python training/eval_seq2seq.py --backend translategemma --model "$MODEL" \
        --adapter "$OUT/adapter" --data data/$split.jsonl \
        >> "$LOG_DIR/eval-translategemma-4b.log" 2>&1
done
echo "=== TranslateGemma-4B fine-tuning and evaluation complete! ==="
