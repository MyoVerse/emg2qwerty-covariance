#!/bin/bash
# Exports the best checkpoints of the original experiment runs (weights only)
# plus the Fréchet means into models/ of this repository, and packs them into
# models.tar.gz for upload as a GitHub release asset.
# Usage: bash scripts/export_release_models.sh /path/to/emg2qwerty
set -euo pipefail
cd "$(dirname "$0")/.."
src=${1:?path to the original emg2qwerty workspace}

best() {  # <log file> -> path of the val/CER-selected checkpoint of that run
  tr '\r' '\n' < "$1" | grep -ao "best_checkpoint': '[^']*" | tail -1 | sed "s/best_checkpoint': '//"
}

mkdir -p models/personalized-randominit models/personalized-finetuned
python scripts/export_checkpoint.py \
  "$src/logs/covariance_znorm_aug_valcer_generic/generic/checkpoints/epoch=142-step=128128.ckpt" \
  models/generic.ckpt
for u in 0 1 2 3 4 5 6 7; do
  python scripts/export_checkpoint.py "$(best $src/logs/covariance_znorm_aug_valcer/user$u.greedy.out)" \
    models/personalized-randominit/user$u.ckpt
  python scripts/export_checkpoint.py "$(best $src/logs/covariance_znorm_aug_valcer_finetune/user$u.greedy.out)" \
    models/personalized-finetuned/user$u.ckpt
done
cp "$src/emg2qwerty_data/frechet_means_znorm_all.pt" models/frechet_means.pt

tar -czvf models.tar.gz models/frechet_means.pt models/generic.ckpt \
  models/personalized-randominit models/personalized-finetuned
ls -la models.tar.gz
