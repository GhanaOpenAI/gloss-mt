#!/usr/bin/env bash
# Remaining controllability runs. OPUS-MT on the Bible now (small); TranslateGemma on the Bible
# after its validation run; NLLB 3.3B (validation + Bible) once the queue reaches the smaller
# NLLB 600M fine-tune, which leaves room on the GPU.
P=/mnt/volume_d2wey28/projects/twi-translation-tool-calling
CK=$P/checkpoints
B="--jsonl $P/repo/build/bible/asante_gnt_300.jsonl"
V="--data $P/repo/build/glossary-chunks-v2/dataset --n 300"
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $P/repo
run() { echo "[$(date "+%F %T")] $1"; $P/.venv-seq2seq/bin/python training/eval_controllability.py --name "$@" >> $P/logs/controllability.log 2>&1; echo "rc=$?"; }
run bible-gloss-mt-opus-mt-en-tw --model $CK/opus-mt-en-tw-twi/final $B
run bible-opus-mt-en-tw-base --model Helsinki-NLP/opus-mt-en-tw $B
until grep -q "done" $P/logs/controllability.out; do sleep 60; done
run bible-gloss-mt-translategemma-4b --backend translategemma --model google/translategemma-4b-it \
    --adapter $CK/translategemma-4b-twi/adapter --batch-size 4 $B
until grep -q "START run_train_nllb_600m.sh" $P/logs/queue-nllb-madlad.out; do sleep 120; done
sleep 600  # let the 600M run pass its first eval (peak memory)
for d in V B; do
    pre=$([ $d = B ] && echo bible- || echo "")
    run ${pre}gloss-mt-nllb-3.3b --model $CK/nllb-3.3b-twi/final --batch-size 8 ${!d}
    run ${pre}nllb-200-3.3b-base --model facebook/nllb-200-3.3B --batch-size 8 ${!d}
done
echo "[$(date "+%F %T")] all-done"
