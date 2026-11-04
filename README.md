# emg2qwerty-covariance

Spatial covariance inputs with causal session recentering for the
[emg2qwerty](https://github.com/facebookresearch/emg2qwerty) benchmark
This is a fork of emg2qwerty that replaces the log-spectrogram front end with sliding-window
covariance matrices recentered at a per-session Fréchet mean, and evaluates
every model with **strictly causal** test-time normalization. The model,
data splits, training recipe and decoders are unchanged from upstream.

## Method

Upstream feeds the TDS-Conv-CTC model a log-spectrogram per electrode
(16 electrodes x 33 frequency bins per band). Here each band's input frame is
instead its 16 x 16 spatial covariance matrix:

1. **Z-normalization** of each electrode over time
   (`ZNormalizeTime` / `CausalZNormalizeTime`).
2. **Sliding covariance** over 64-sample windows with a 16-sample hop (the
   spectrogram's `n_fft`/`hop_length`, so the frame rate is the same 125 Hz) (`SlidingCovariance`).
3. **Session recentering** `M^{-1/2} P M^{-1/2}` at the session's
   Log-Cholesky Fréchet mean `M` (Lin, 2019), an unsupervised,
   label-free session adaptation (`SessionRecentering` /
   `CausalSessionRecentering`).
4. **SPDAugment** during training: the covariance analogue of SpecAugment,
   masking time frames and electrode blocks to the identity (the session
   mean) so matrices stay SPD.

Training and validation windows (4 s) use the offline statistics: each window
is z-normalized on its own and recentered at its session's precomputed mean
(`scripts/compute_frechet_means.py`). At **test time**, sessions are fed
whole and both the z-normalization and the Fréchet mean are replaced by
running (expanding-window) estimates that use no future samples, so the
reported test CER is what a streaming deployment achieves. No means file is
needed at test time. See [config/transforms/covariance.yaml](config/transforms/covariance.yaml).

## Results

Test CER (%) on the 8 benchmark users with causal test-time normalization.
Upstream log-spectrogram numbers are from Table 2 of the emg2qwerty paper
([scripts/experimental_results.py](scripts/experimental_results.py)).
Beam search uses the upstream 6-gram character LM and decoder defaults
(beam 50, LM weight 2.0, insertion bonus 2.0, untuned). Per-user numbers and
the beam-search predictions are in [results/](results/).

| Model | Greedy, spectrogram (upstream) | Greedy, covariance | Beam + LM, spectrogram (upstream) | Beam + LM, covariance |
|---|---|---|---|---|
| Generic (96 users), zero-shot | 55.38 | **50.49** | 51.78 | **44.56** |
| Personalized, random init | 15.38 | **12.85** | 9.55 | **6.97** |
| Personalized, fine-tuned from generic | 11.28 | **9.90** | 6.95 | **5.40** |

The covariance front end lowers the mean test CER in every setting (6 or 7 of
8 users, depending on the row). Covariance models trained from scratch
reach parity with upstream's fine-tuned models under the LM (6.97 vs 6.95).

## Setup

```shell
git clone https://github.com/<github-user>/emg2qwerty-covariance.git ~/emg2qwerty-covariance
cd ~/emg2qwerty-covariance
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .

# Download the dataset, extract, and symlink to ~/emg2qwerty-covariance/data
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty-covariance/data
```

Pretrained checkpoints and the per-session Fréchet means are distributed as
[release assets](https://github.com/<github-user>/emg2qwerty-covariance/releases)
rather than through git:

```shell
cd ~/emg2qwerty-covariance
wget -qO- https://github.com/<github-user>/emg2qwerty-covariance/releases/download/v1.0/models.tar.gz | tar -xzv
```

This populates:

```
models/
  frechet_means.pt                 # Per-session Fréchet means (train/val sessions)
  generic.ckpt                     # Generic model, 96 training users
  personalized-randominit/user{0..7}.ckpt
  personalized-finetuned/user{0..7}.ckpt
  lm/wikitext-103-6gram-charlm.bin # Upstream 6-gram character LM (tracked in git)
```

## Fréchet means

Training (and validation during training) needs the Log-Cholesky Fréchet
mean of every train/val session, computed exactly as the model sees the data
(per 4 s window, z-normalized). To regenerate `models/frechet_means.pt` for
all sessions used by the generic and personalized models:

```shell
python scripts/compute_frechet_means.py \
  --user-config config/user/generic.yaml \
  --user-config "config/user/user[0-7].yaml" \
  --num-workers 16
```

The covariance options of the script must match
`config/transforms/covariance.yaml` (they do by default).

## Training

Covariance inputs are the default config (`transforms=covariance`,
`model=tds_conv_ctc_covariance`); pass `transforms=log_spectrogram
model=tds_conv_ctc` to train the upstream spectrogram baseline instead.
Checkpoints are selected by `val/CER`, as upstream.

Generic user model (effective batch size 256, as upstream's 8 GPUs):

```shell
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=2 \
  +trainer.strategy=ddp +trainer.accumulate_grad_batches=4 \
  num_workers=16
```

Personalized user models from random initialization:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.accelerator=gpu trainer.devices=1 \
  --multirun
```

Personalized user models fine-tuned from the generic model:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  trainer.accelerator=gpu trainer.devices=1 \
  checkpoint="${HOME}/emg2qwerty-covariance/models/generic.ckpt" \
  --multirun
```

If you are using a Slurm cluster, include the `cluster=slurm` override to pick
up `config/cluster/slurm.yaml` (Hydra Submitit launcher).

To export a training checkpoint for distribution (weights and
hyper-parameters only):

```shell
python scripts/export_checkpoint.py logs/<run>/checkpoints/<best>.ckpt models/generic.ckpt
```

## Testing

Test sessions are always evaluated with the causal transforms. Greedy
decoding of the fine-tuned models:

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty-covariance/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=gpu \
  decoder=ctc_greedy \
  --multirun
```

Beam-search decoding with the 6-gram character LM (CPU-bound):

```shell
python -m emg2qwerty.train \
  user="glob(user*)" \
  checkpoint="${HOME}/emg2qwerty-covariance/models/personalized-finetuned/\${user}.ckpt" \
  train=False trainer.accelerator=cpu \
  decoder=ctc_beam \
  --multirun
```

Note that `val/CER` printed by these runs is computed on 4 s windows with the
offline transforms (it is the checkpoint-selection metric); `test/CER` is the
whole-session causal result.

To dump the decoded test predictions of a checkpoint (one JSON per user, with
its test metrics), use `scripts/dump_predictions.py` with the same overrides;
`scripts/dump_all_predictions.sh` does so for every released checkpoint and
both decoders, and `scripts/collect_results.py` tabulates the dumps into
[results/test_cer.md](results/test_cer.md). `scripts/compare_predictions.py`
compares two dumps character by character with utterance-level paired tests,
e.g., against the upstream spectrogram models whose predictions are included
under `results/predictions/baseline_*`:

```shell
python scripts/compare_predictions.py \
  --model-a results/predictions/finetuned_beam \
  --model-b results/predictions/baseline_finetuned_beam
```

The 6-gram character-level language model under `models/lm/` is the upstream
one (built from WikiText-103 with KenLM; see `scripts/lm/`).

## License

emg2qwerty-covariance is CC-BY-NC-4.0 licensed, as found in the LICENSE file,
like the upstream emg2qwerty it derives from.

## Citing

If you use this code, please cite the emg2qwerty dataset and baselines:

```
@inproceedings{emg2qwerty_neurips2024,
 author = {Sivakumar, Viswanath and Seely, Jeffrey and Du, Alan and Bittner, Sean R and Berenzweig, Adam and Bolarinwa, Anuoluwapo and Gramfort, Alexandre and Mandel, Michael I},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {91373--91389},
 publisher = {Curran Associates, Inc.},
 title = {emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf},
 volume = {37},
 year = {2024}
}
```

and the Log-Cholesky metric used for the session means:

```
@article{lin2019riemannian,
 author = {Lin, Zhenhua},
 title = {Riemannian Geometry of Symmetric Positive Definite Matrices via Cholesky Decomposition},
 journal = {SIAM Journal on Matrix Analysis and Applications},
 volume = {40},
 number = {4},
 pages = {1353--1370},
 year = {2019}
}
```
