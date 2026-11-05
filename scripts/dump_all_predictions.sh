#!/bin/bash
# Dumps the test-set predictions of all released covariance checkpoints under
# causal test-time normalization, for both decoders, into
# results/predictions/<model>_<decoder>/user*.json, then tabulates them with
# scripts/collect_results.py. Run from the repository root with the
# emg2qwerty conda env activated, and the dataset at ./data.
#
# Greedy runs sequentially on the GPU (a few minutes in total). Beam search is
# CPU-bound, so all 24 beam jobs run in parallel on the CPU (about an hour).
#
# The validation sessions are also decoded greedily the same way (whole
# sessions, causal normalization) into results/predictions/val/, by letting
# them stand in for the test split, and tabulated into results/val_cer.md.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/dump

checkpoint() {  # <model> <user>
  case $1 in
    generic) echo models/generic.ckpt ;;
    randominit) echo models/personalized-randominit/$2.ckpt ;;
    finetuned) echo models/personalized-finetuned/$2.ckpt ;;
  esac
}

for model in generic randominit finetuned; do
  for u in 0 1 2 3 4 5 6 7; do
    python scripts/dump_predictions.py user=user$u checkpoint="$(checkpoint $model user$u)" \
      decoder=ctc_greedy +predictions_path=results/predictions/${model}_greedy/user$u.json \
      hydra.run.dir=logs/dump/${model}_greedy/user$u
  done
done

export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
for model in generic randominit finetuned; do
  for u in 0 1 2 3 4 5 6 7; do
    python scripts/dump_predictions.py user=user$u checkpoint="$(checkpoint $model user$u)" \
      decoder=ctc_beam num_workers=1 +predictions_path=results/predictions/${model}_beam/user$u.json \
      hydra.run.dir=logs/dump/${model}_beam/user$u > logs/dump/${model}_beam_user$u.log 2>&1 &
  done
done
wait

python scripts/collect_results.py

for model in generic randominit finetuned; do
  for u in 0 1 2 3 4 5 6 7; do
    python scripts/dump_predictions.py user=user$u checkpoint="$(checkpoint $model user$u)" \
      decoder=ctc_greedy 'dataset.test=${dataset.val}' \
      +predictions_path=results/predictions/val/${model}_greedy/user$u.json \
      hydra.run.dir=logs/dump/val/${model}_greedy/user$u
  done
done

python scripts/collect_results.py --split val
