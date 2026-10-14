#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="facebook/nllb-200-3.3B"
OUT="$PROJ/checkpoints/nllb-3.3b-twi-lora-gold"
VENV="$PROJ/.venv-seq2seq"  # .venv-train packages (via .pth) + spaCy
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== [1/2] LoRA fine-tune (r=32) of facebook/nllb-200-3.3B (eng_Latn -> twi_Latn, gold-consistent pretranslated training terms, best by composite of BLEU/chrF2/loss; best + last checkpoints kept) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
# GEMINI_API_KEY: GeminiScore is reported in the final eval only (not used to pick checkpoints)
[ -f "$PROJ/.env" ] && set -a && . "$PROJ/.env" && set +a
cd "$PROJ/repo"

$VENV/bin/python training/train_seq2seq.py \
    --model "$MODEL" \
    --data data \
    --out "$OUT" \
    --chunk-tokens 192 \
    --max-src 1024 \
    --epochs 30 \
    --early-stopping-patience 3 \
    --lora-r 32 \
    --train-glossary gold \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 2e-4 \
    --eval-steps 1000 \
    >> "$LOG_DIR/train-nllb-3.3b-lora-gold.log" 2>&1

echo "=== [2/2] Evaluating on validation and test ==="
for split in validation test; do
    $VENV/bin/python training/eval_seq2seq.py --model "$OUT/final" --data data/$split.jsonl \
        >> "$LOG_DIR/eval-nllb-3.3b-lora-gold.log" 2>&1
done
echo "=== NLLB-3.3B LoRA (gold terms) fine-tuning and evaluation complete! ==="
