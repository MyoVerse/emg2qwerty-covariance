# Test CER (%)

Covariance models are evaluated with strictly causal test-time normalization (config/transforms/covariance.yaml); the spectrogram baselines are the upstream emg2qwerty results (Table 2 of the paper).

## No LM (greedy)

| | user0 | user1 | user2 | user3 | user4 | user5 | user6 | user7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Generic, zero-shot, covariance | 50.60 | 45.50 | 43.63 | 58.85 | 60.70 | 44.69 | 53.49 | 46.45 | 50.49 |
| Generic, zero-shot, spectrogram (upstream) | 61.48 | 59.96 | 48.00 | 54.69 | 58.24 | 53.86 | 54.66 | 52.17 | 55.38 |
| Personalized, random init, covariance | 18.81 | 10.84 | 8.34 | 11.03 | 8.54 | 6.82 | 20.56 | 17.87 | 12.85 |
| Personalized, random init, spectrogram (upstream) | 26.60 | 14.15 | 11.12 | 13.25 | 10.51 | 7.65 | 18.82 | 20.94 | 15.38 |
| Personalized, fine-tuned from generic, covariance | 15.01 | 8.98 | 7.25 | 7.62 | 7.32 | 5.42 | 15.22 | 12.36 | 9.90 |
| Personalized, fine-tuned from generic, spectrogram (upstream) | 20.57 | 10.32 | 8.41 | 8.93 | 7.91 | 5.81 | 14.21 | 14.06 | 11.28 |

## LM (beam)

| | user0 | user1 | user2 | user3 | user4 | user5 | user6 | user7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Generic, zero-shot, covariance | 44.81 | 38.93 | 36.33 | 56.21 | 56.57 | 32.90 | 51.00 | 39.75 | 44.56 |
| Generic, zero-shot, spectrogram (upstream) | 58.40 | 56.66 | 42.90 | 51.63 | 54.23 | 49.55 | 52.59 | 48.32 | 51.78 |
| Personalized, random init, covariance | 11.07 | 5.48 | 4.70 | 4.87 | 3.93 | 2.82 | 12.82 | 10.08 | 6.97 |
| Personalized, random init, spectrogram (upstream) | 20.72 | 7.96 | 6.78 | 7.29 | 4.85 | 3.82 | 11.66 | 13.28 | 9.55 |
| Personalized, fine-tuned from generic, covariance | 8.96 | 4.52 | 4.37 | 3.53 | 3.68 | 2.19 | 9.38 | 6.55 | 5.40 |
| Personalized, fine-tuned from generic, spectrogram (upstream) | 15.04 | 6.18 | 5.08 | 4.75 | 3.91 | 3.16 | 8.83 | 8.68 | 6.95 |
