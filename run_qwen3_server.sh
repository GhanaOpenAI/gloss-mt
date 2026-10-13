#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
VENV="$PROJ/.venv"
MODEL="$PROJ/checkpoints/qwen3-0.6b-twi-tool/merged"
LOG_DIR="$PROJ/logs"
mkdir -p "$LOG_DIR"

echo "Starting vLLM Qwen 3 on port 8301..."
pkill -f "port 8301" || true
sleep 1

nohup $VENV/bin/vllm serve "$MODEL" \
    --served-model-name twi-translator-qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8301 \
    --host 127.0.0.1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.08 \
    > "$LOG_DIR/vllm-qwen3.log" 2>&1 &

echo "Waiting for vLLM on port 8301 to become ready..."
until curl -s http://127.0.0.1:8301/v1/models | grep -q 'twi-translator-qwen3'; do
    sleep 2
done
echo "vLLM on port 8301 is ready!"
