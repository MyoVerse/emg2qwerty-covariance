# Validation CER (%)

`val/CER` of `python -m emg2qwerty.train train=False` on the released checkpoints: the validation sessions in 4 s windows under the non-causal val transforms of config/transforms/covariance.yaml (per-window z-normalization, recentering at the precomputed session Fréchet mean of models/frechet_means.pt), i.e., the metric that selected the checkpoints, as upstream. The spectrogram baselines are the upstream Val CER (Table 2 of the paper). Test sessions are evaluated causally instead, see results/test_cer.md.

## No LM (greedy)

| | user0 | user1 | user2 | user3 | user4 | user5 | user6 | user7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Generic, zero-shot, covariance | 45.50 | 46.27 | 47.06 | 57.83 | 60.57 | 49.67 | 57.88 | 46.17 | 51.37 |
| Generic, zero-shot, spectrogram (upstream) | 60.07 | 55.59 | 47.38 | 59.03 | 58.93 | 56.01 | 58.08 | 49.45 | 55.57 |
| Personalized, random init, covariance | 15.45 | 9.48 | 9.44 | 10.98 | 8.73 | 8.63 | 21.74 | 18.56 | 12.88 |
| Personalized, random init, spectrogram (upstream) | 24.13 | 11.31 | 10.05 | 14.61 | 10.64 | 9.36 | 22.20 | 22.91 | 15.65 |
| Personalized, fine-tuned from generic, covariance | 13.37 | 7.52 | 7.29 | 7.48 | 7.61 | 6.87 | 17.25 | 13.18 | 10.07 |
| Personalized, fine-tuned from generic, spectrogram (upstream) | 17.96 | 8.39 | 8.16 | 9.54 | 7.58 | 7.15 | 17.17 | 15.19 | 11.39 |
