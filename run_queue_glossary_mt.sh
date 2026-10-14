#!/usr/bin/env bash
# Unattended queue for the glossary-conditioned MT experiments, one job at a time on the GPU:
#   zero-shot baselines -> NLLB-3.3B fine-tune + eval -> TranslateGemma-4B fine-tune + eval.
# A failed job is logged and the queue moves on. Pass runner names to run a subset, e.g.
#   ./run_queue_glossary_mt.sh run_train_nllb_3.3b_lora.sh run_train_translategemma_4b.sh
PROJ="/mnt/volume_d2wey28/projects/twi-translation-tool-calling"
LOG="$PROJ/logs/queue-glossary-mt.log"
cd "$PROJ"

run() {
    echo "[$(date '+%F %T')] START $1" >> "$LOG"
    bash "$PROJ/$1" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] END $1 rc=$?" >> "$LOG"
}
JOBS=("$@")
[ ${#JOBS[@]} -gt 0 ] || JOBS=(run_zeroshot_baselines.sh run_train_nllb_3.3b_lora.sh run_train_translategemma_4b.sh)
for job in "${JOBS[@]}"; do
    run "$job"
done
echo "[$(date '+%F %T')] QUEUE DONE" >> "$LOG"
