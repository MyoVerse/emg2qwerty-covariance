# Validation CER (%)

Validation sessions of each user decoded exactly like the test sessions: whole sessions fed at once under strictly causal normalization (config/transforms/covariance.yaml), from the released checkpoints. This is not the windowed, non-causal `val/CER` logged during training, which only selects checkpoints (see results/test_cer.md for the test sessions).

## No LM (greedy)

| | user0 | user1 | user2 | user3 | user4 | user5 | user6 | user7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Generic, zero-shot, covariance | 44.90 | 46.47 | 47.59 | 56.94 | 60.42 | 49.19 | 57.59 | 46.11 | 51.15 |
| Personalized, random init, covariance | 15.48 | 8.93 | 8.32 | 9.67 | 7.91 | 8.11 | 24.11 | 18.44 | 12.62 |
| Personalized, fine-tuned from generic, covariance | 13.19 | 7.03 | 6.56 | 6.84 | 6.81 | 6.62 | 18.97 | 12.83 | 9.86 |
