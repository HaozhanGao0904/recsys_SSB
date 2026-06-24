# Head-Weight PL Estimation Report

## Identification Assumption

Because the true candidate set is unobserved, PL does not estimate exposure against unrecommended videos. It estimates the score weights that best rationalize the observed order conditional on exposure.

Public descriptions of modern recommender systems suggest candidates are often ranked by learned value or engagement scores, but final order may also reflect reranking, exploration, diversity, integrity, and business rules. Therefore PL should be interpreted as a noisy approximation to the platform ranking stage, not the exact serving rule.

## Data

- Observed validation rows used: 899,833
- Validation sessions used: 46,940
- Users used: 6,791
- Sessions dropped because K_s < 2: 8,619

No test sessions and no unrecommended sampled videos are used.

## Objective

Plackett-Luce observed-order loss with session weighting.

- tau: 1.0
- lambda_l2: 0.001
- session weighting: session
- primary fit: unconstrained

## Primary Unconstrained Weights

```json
{
  "w_complete": 0.05088338408773398,
  "w_long": 0.014799267862244149,
  "w_rewatch": -0.05711003386925411,
  "w_neg": -0.3119390730200324
}
```

## Constrained Diagnostic Weights

```json
{
  "w_complete": 0.041181241473535475,
  "w_long": 0.0,
  "w_rewatch": 0.0,
  "w_neg": -0.3083114442776101
}
```

The unconstrained weights are now the primary weights saved for downstream use. The constrained weights are retained only as a diagnostic.

## Empirical Conclusion

```json
{
  "earlier_videos_have_higher_equal_signed_score_on_average": true,
  "constrained_weights_sensible_by_construction": true,
  "unconstrained_weights_sensible": false,
  "pl_constrained_beats_random_baseline": true
}
```

## Main Diagnostics

PL metrics, pairwise order metrics, Spearman correlations, and mean heads by rank are saved in `python_code_new/outputs/head_weight_pl_metrics.json` and `python_code_new/outputs/pl_mean_heads_by_rank.csv`.
