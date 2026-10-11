#!/usr/bin/env bash
# Automatically monitors the expanded dictionary benchmark on H200,
# syncs the results, pushes them to GitHub nsanku-mt-benchmark,
# regenerates the paper comparison tables, and starts Gemma 4B fine-tuning.

set -e

BENCH_LOCAL="/home/owusus/Documents/GitHub/nsanku-mt-benchmark"
REPO_LOCAL="/home/owusus/Documents/GitHub/translation-tool-calling-dataset"

echo "=== Monitoring Expanded Dictionary Benchmark on H200 ==="

while true; do
    status=$(ssh -o ConnectTimeout=10 h200 "grep -q '=== Expanded Dictionary Benchmark Complete! ===' /mnt/volume_d2wey28/projects/nsanku-mt-benchmark/bench_expanded_dict.log 2>/dev/null && echo 'DONE' || echo 'RUNNING'")
    if [ "$status" = "DONE" ]; then
        echo "Benchmark on H200 has completed!"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Still benchmarking on H200... waiting 30s"
    sleep 30
done

echo "=== Syncing benchmark outputs from H200 ==="
rsync -az h200:/mnt/volume_d2wey28/projects/nsanku-mt-benchmark/translations/twi-translator-gemma4-31b-expanded-dict/ "$BENCH_LOCAL/translations/twi-translator-gemma4-31b-expanded-dict/"
rsync -az h200:/mnt/volume_d2wey28/projects/nsanku-mt-benchmark/benchmarks/twi_asante.yaml "$BENCH_LOCAL/benchmarks/twi_asante.yaml"

cd "$BENCH_LOCAL"
echo "=== Assembling benchmark scores locally ==="
python3 run_benchmark.py --assemble

echo "=== Pushing benchmark results to GitHub GhanaNLP/nsanku-mt-benchmark ==="
git add benchmarks/ space/ translations/
git commit -m "feat(benchmarks): add twi-translator-gemma4-31b-expanded-dict complete benchmark scores" || true
git push origin main || true

echo "=== Updating paper comparison tables in project repo ==="
cd "$REPO_LOCAL"
python3 scripts/generate_paper_results.py

git add results/
git commit -m "docs(results): update paper comparison tables with expanded dictionary scores" || true
git push origin main || true

echo "=== Launching Gemma 4B (15 epochs, early stopping) fine-tuning on H200 ==="
ssh h200 "pkill -f 'vllm serve' || true; cd /mnt/volume_d2wey28/projects/twi-translation-tool-calling && nohup ./run_train_e4b.sh > logs/train-e4b-startup.log 2>&1 < /dev/null &"

echo "=== Pipeline finished successfully! ==="
