#!/usr/bin/env bash
set -e

PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
MODEL="Qwen/Qwen3-0.6B"
OUT="$PROJ/checkpoints/qwen3-0.6b-twi-tool"
MERGED="$OUT/merged"
VENV="$PROJ/.venv-train"
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR" "$OUT"

echo "=== [1/3] SFT Fine-Tuning Qwen/Qwen3-0.6B (LoRA r=32, bf16, epochs=5, early stopping patience=8) ==="
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ/repo"

$VENV/bin/python training/train_sft.py \
    --model "$MODEL" \
    --data data \
    --out "$OUT" \
    --epochs 5 \
    --early-stopping-patience 8 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 2e-4 \
    --lora-r 32 \
    --eval-steps 150 \
    --backend hf \
    >> "$LOG_DIR/train-qwen3.log" 2>&1

echo "=== [2/3] Merging best LoRA adapter into bf16 standalone weights ==="
$VENV/bin/python training/merge_lora.py \
    --base "$MODEL" \
    --adapter "$OUT/adapter" \
    --out "$MERGED"

echo "=== [3/3] Uploading to Hugging Face Hub: ghanaopenai/gloss-mt-qwen3-0.6b ==="
$PROJ/.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('ghanaopenai/gloss-mt-qwen3-0.6b', exist_ok=True, private=False)
api.upload_folder(
    folder_path='$MERGED',
    repo_id='ghanaopenai/gloss-mt-qwen3-0.6b',
    repo_type='model',
    commit_message='feat: release fine-tuned Qwen 3 0.6B GLOSS-MT Twi translator with tool calling'
)
print('Uploaded ghanaopenai/gloss-mt-qwen3-0.6b successfully!')
"
echo "=== Fine-tuning and export of Qwen 3 0.6B complete! ==="
