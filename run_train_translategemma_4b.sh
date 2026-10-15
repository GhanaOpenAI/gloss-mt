#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="google/translategemma-4b-it"
OUT="$PROJ/checkpoints/translategemma-4b-twi"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== LoRA fine-tune of google/translategemma-4b-it (r=32, glossary-conditioned, best by composite of BLEU/chrF2/loss; best + last checkpoints kept) ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
# GEMINI_API_KEY: GeminiScore is reported in the final eval only (not used to pick checkpoints)
[ -f "$PROJ/.env" ] && set -a && . "$PROJ/.env" && set +a
cd "$PROJ/repo"

$VENV/bin/python training/train_translategemma.py \
    --model "$MODEL" \
    --out "$OUT" \
    --data $PROJ/repo/build/glossary-chunks-v2/dataset \
    --epochs 30 \
    --early-stopping-patience 5 \
    --batch-size 8 \
    --grad-accum 4 \
    --lr 2e-4 \
    --lora-r 32 \
    --eval-steps 1000 \
    >> "$LOG_DIR/train-translategemma-4b.log" 2>&1

echo "=== TranslateGemma-4B fine-tuning complete! ==="
