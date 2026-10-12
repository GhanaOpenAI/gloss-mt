#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="/mnt/volume_d2wey28/models/gemma-4-E4B-it"
OUT="$PROJ/checkpoints/gemma4-e4b-twi-tool"
MERGED="$OUT/merged"
VENV="$PROJ/.venv-train"
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== [1/3] SFT Fine-Tuning google/gemma-4-E4B-it (LoRA r=32, bf16, epochs=15, early stopping patience=10) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
cd "$PROJ/repo"

$VENV/bin/python training/train_sft.py \
    --model "$MODEL" \
    --data data \
    --out "$OUT" \
    --epochs 15 \
    --early-stopping-patience 10 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --lora-r 32 \
    --eval-steps 150 \
    --backend hf \
    >> "$LOG_DIR/train-e4b.log" 2>&1

echo "=== [2/3] Merging best LoRA adapter into bf16 standalone weights ==="
$VENV/bin/python training/merge_lora.py \
    --base "$MODEL" \
    --adapter "$OUT/adapter" \
    --out "$MERGED"

echo "=== [3/3] Uploading to Hugging Face Hub: ghanaopenai/gloss-mt ==="
$PROJ/.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('ghanaopenai/gloss-mt', exist_ok=True, private=False)
api.upload_folder(
    folder_path='$MERGED',
    repo_id='ghanaopenai/gloss-mt',
    repo_type='model',
    commit_message='feat: release fine-tuned Gemma 4 E4B GLOSS-MT Twi translator with tool calling'
)
print('Uploaded ghanaopenai/gloss-mt successfully!')
"
echo "=== Fine-tuning and export of Gemma 4 E4B complete! ==="
