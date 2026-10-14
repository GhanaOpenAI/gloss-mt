#!/usr/bin/env bash
# Zero-shot (no fine-tuning, no glossary) test-set baselines for the seq2seq-pipeline candidates.
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
VENV="$PROJ/.venv-seq2seq"
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

echo "=== Zero-shot NLLB-200 3.3B (eng_Latn -> twi_Latn) ==="
$VENV/bin/python training/eval_seq2seq.py --model facebook/nllb-200-3.3B --no-glossary \
    --data data/test.jsonl >> "$LOG_DIR/eval-zeroshot.log" 2>&1

echo "=== Zero-shot TranslateGemma 4B (en -> ak) ==="
$VENV/bin/python training/eval_seq2seq.py --backend translategemma --model google/translategemma-4b-it \
    --no-glossary --data data/test.jsonl >> "$LOG_DIR/eval-zeroshot.log" 2>&1
echo "=== Zero-shot baselines complete! ==="
