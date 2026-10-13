#!/usr/bin/env bash
set -e

BENCH_DIR="/mnt/volume_d2wey28/projects/nsanku-mt-benchmark"
VENV="$BENCH_DIR/.venv"
LOG_FILE="$BENCH_DIR/bench_qwen3.log"

echo "=== [1/3] Ensuring vLLM on port 8301 is active ===" >> "$LOG_FILE"
until curl -s http://127.0.0.1:8301/v1/models | grep -q 'twi-translator-qwen3'; do
    echo "Starting vLLM on port 8301..." >> "$LOG_FILE"
    /mnt/volume_d2wey28/projects/twi-translation-tool-calling/run_qwen3_server.sh >> "$LOG_FILE" 2>&1
    sleep 3
done
echo "vLLM on port 8301 is online." >> "$LOG_FILE"

echo "=== [2/3] Running official 84-document benchmark for gloss-mt-qwen3-0.6b ===" >> "$LOG_FILE"
cd "$BENCH_DIR"
$VENV/bin/python run_benchmark.py \
    --models gloss-mt-qwen3-0.6b \
    --langs twi_asante \
    --directions from_en \
    >> "$LOG_FILE" 2>&1

echo "=== [3/3] Assembling document-level scores into benchmarks/twi_asante.yaml ===" >> "$LOG_FILE"
$VENV/bin/python run_benchmark.py --assemble >> "$LOG_FILE" 2>&1

echo "=== Qwen 3 0.6B Benchmark Complete! ===" >> "$LOG_FILE"
