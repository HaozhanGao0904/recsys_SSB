# Head-Weight Pairwise Estimation Report

Input pair deltas: `/Users/haozhangao/Desktop/RecSys Research/python_code_new/outputs/validation_pair_head_deltas.parquet`

## Sample

- Pairs: 3,778,270
- Sessions: 55,559
- Users: 6,809
- Positive session-videos: 755,654
- Negative session-videos: 3,706,944
- Negative exclusion policies: {'user_never_recommended_full_panel': 3778270}
- Future same-user recommendations sampled as negatives: 0
- Ever same-user recommendations sampled as negatives: 0

## Objective

Convex Bradley-Terry pairwise logistic loss with L2 penalty.

- temperature: 1.0
- lambda_l2: 0.0001
- pair weighting: `pair`
- primary fit: `constrained`

## Primary Weights

```json
{
  "w_complete": 0.0,
  "w_long": 0.0,
  "w_rewatch": 0.0,
  "w_neg": -0.7563880146420972
}
```

## Comparison

```json
[
  {
    "model": "estimated_unconstrained",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.5850008072477615,
    "pairwise_auc": 0.6304697206357901,
    "pairwise_log_loss": 0.6629209758384457,
    "mean_margin": 0.12503260011620537,
    "median_margin": 0.045665292710107025
  },
  {
    "model": "estimated_constrained",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.48701336855227395,
    "pairwise_auc": 0.49811093814489943,
    "pairwise_log_loss": 0.6928707861759199,
    "mean_margin": 0.0008773828785260412,
    "median_margin": -0.00014697461889584285
  },
  {
    "model": "single_neg_negative_weight",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.48701336855227395,
    "pairwise_auc": 0.49811093814489943,
    "pairwise_log_loss": 0.6928506606998199,
    "mean_margin": 0.0011599640152167078,
    "median_margin": -0.00019431114196777344
  },
  {
    "model": "single_long",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.44415777591331485,
    "pairwise_auc": 0.40505121580887554,
    "pairwise_log_loss": 0.715536839718867,
    "mean_margin": -0.03603063586950204,
    "median_margin": -0.003583047306165099
  },
  {
    "model": "single_complete",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.4243280654902905,
    "pairwise_auc": 0.39455175927880637,
    "pairwise_log_loss": 0.7375820004171064,
    "mean_margin": -0.061029459153771656,
    "median_margin": -0.026171937584877014
  },
  {
    "model": "equal_signed",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.42441858310814207,
    "pairwise_auc": 0.38469737988215713,
    "pairwise_log_loss": 0.8038607298124051,
    "mean_margin": -0.13267540356298507,
    "median_margin": -0.05598956858739257
  },
  {
    "model": "single_rewatch",
    "num_pairs": 3778270,
    "pairwise_accuracy": 0.4181935118453684,
    "pairwise_auc": 0.3785233886280155,
    "pairwise_log_loss": 0.7152563456836556,
    "mean_margin": -0.03677527255492805,
    "median_margin": -0.0026499145897105336
  }
]
```

## Score Formula

$$
\operatorname{score}
= w_{	ext{complete}}p_{	ext{complete}}
+ w_{	ext{long}}p_{	ext{long}}
+ w_{	ext{rewatch}}p_{	ext{rewatch}}
+ w_{	ext{neg}}p_{	ext{neg}}.
$$
