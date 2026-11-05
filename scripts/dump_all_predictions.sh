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
# Finally, the windowed, non-causal val/CER of every checkpoint (the metric
# that selects checkpoints, as upstream) is recorded by emg2qwerty.train
# (train=False, greedy) into results/metrics/<model>_greedy/user*.json and
# tabulated into results/val_cer.md.
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

unset CUDA_VISIBLE_DEVICES
for model in generic randominit finetuned; do
  for u in 0 1 2 3 4 5 6 7; do
    python -m emg2qwerty.train user=user$u checkpoint="$(checkpoint $model user$u)" \
      train=False trainer.accelerator=gpu trainer.devices=1 decoder=ctc_greedy \
      +metrics_path=results/metrics/${model}_greedy/user$u.json \
      hydra.run.dir=logs/dump/metrics/${model}_greedy/user$u
  done
done

python scripts/collect_results.py --split val
