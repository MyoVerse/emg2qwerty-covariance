# emg2qwerty-covariance

Spatial covariance matrix input for the
[emg2qwerty](https://github.com/facebookresearch/emg2qwerty) benchmark.
This is a fork of emg2qwerty that replaces the log-spectrogram front end with sliding-window
covariance matrices. The model,
data splits, training recipe and decoders are unchanged from upstream.

## Method

Upstream feeds the TDS-Conv-CTC model a log-spectrogram per electrode
(16 electrodes x 33 frequency bins per band). Here each band's input frame is
instead its 16 x 16 spatial covariance matrix:

## Results

Test CER (%) on the 8 benchmark users.
Upstream log-spectrogram numbers are from Table 2 of the emg2qwerty paper
([scripts/experimental_results.py](scripts/experimental_results.py)).
Beam search uses the upstream 6-gram character LM. Per-user numbers and
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
git clone https://github.com/MyoVerse/emg2qwerty-covariance.git ~/emg2qwerty-covariance
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
[release assets](https://github.com/MyoVerse/emg2qwerty-covariance/releases)
rather than through git:

```shell
cd ~/emg2qwerty-covariance
wget -qO- https://github.com/MyoVerse/emg2qwerty-covariance/releases/download/v1.0/models.tar.gz | tar -xzv
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

## Fréchet mean

Training (and validation during training) needs the Log-Cholesky Fréchet
mean of every train/val session. To regenerate `models/frechet_means.pt` for
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

## Citing

If you use this code, please cite the blog post describing this work:

```
@misc{gowda2026emg2qwerty,
 author = {Gowda, Harsha},
 title = {emg2qwerty: Covariance vs. Spectrogram Representations},
 howpublished = {\url{https://harshavardhanatg.github.io/blog/emg2qwerty.html}},
 year = {2026}
}
```

and the emg2qwerty dataset and baselines:

```
@inproceedings{emg2qwerty_neurips2024,
 author = {Sivakumar, Viswanath and Seely, Jeffrey and Du, Alan and Bittner, Sean R and Berenzweig, Adam and Bolarinwa, Anuoluwapo and Gramfort, Alexandre and Mandel, Michael I},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf},
 year = {2024}
}
```
