#!/usr/bin/env bash
# Fine-tunes in turn: NLLB-200 1.3B, NLLB-200 distilled 600M, MADLAD-400 3B (each logs to logs/train-*.log).
cd /mnt/volume_d2wey28/projects/twi-translation-tool-calling
for s in run_train_nllb_1.3b.sh run_train_nllb_600m.sh run_train_madlad_3b.sh; do
    echo "[$(date "+%F %T")] START $s"
    ./$s
    echo "[$(date "+%F %T")] END $s rc=$?"
done
