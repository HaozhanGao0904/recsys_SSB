# KuaiRand Random-Exposure Validation Notebooks

This folder contains the KuaiRand transfer-validation notebooks used to compare the debiased EU ranker against head-based recommenders on random-exposure interactions.

The analysis uses KuaiRand random recommendations from the test period and keeps the subset closest to the target one-video autoplay UI:

- keep random rows with `is_singleton_user_time == 1`;
- drop rows with `is_click == 1`;
- rank the remaining random rows globally by each recommender score.

## Notebook Order

| Notebook | Role |
|---|---|
| `01_test_period_summary_stats.ipynb` | Inspect KuaiRand test-period logs, tab/policy shares, timestamp block sizes, random-row shares, video-duration distributions, and watch-ratio distributions. |
| `02_random_prediction_features.ipynb` | Build KuaiRand random-row feature tables compatible with the KuaiRec-trained transfer models. |
| `03_pretrain_kuairec_head_mlp_for_kuairand.ipynb` | Train the four-head MLP on KuaiRec using only features available for KuaiRand transfer. |
| `04_predict_kuairand_eu_labels.ipynb` | Train the scalar EU MLP on KuaiRec posterior EU labels using the KuaiRand-compatible feature set. |
| `05_kuairand_random_model_comparison.ipynb` | Score KuaiRand random rows, restrict to singleton/no-click rows, and compare top-ranked engagement metrics for EU, naive head-weight, and completion-only recommenders. |

## Final Reporting Specification

The current reporting table in notebook `05` uses:

- top 10% and top 20% rows;
- metrics: `is_follow`, `is_hate`, `is_like`, `is_profile_enter`, `play_time_ms`, and `watch_ratio`;
- recommenders: predicted EU, naive head score, and completion-only ranking;
- no bootstrap confidence intervals in the final reported table.

The naive head score is:

```text
pred_p_complete + pred_p_long + pred_p_rewatch - pred_p_neg
```

These notebooks assume the local full KuaiRand and KuaiRec data artifacts are restored in the paths documented in the project workspace. The raw KuaiRand logs and large intermediate feature tables are not committed to this GitHub repository.
