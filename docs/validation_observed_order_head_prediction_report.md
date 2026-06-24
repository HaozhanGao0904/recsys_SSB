# Validation Observed-Order Head Prediction Report

## Data

- Run mode: full
- Validation sessions used: 55,559
- Validation observed rows: 908,452
- Sessions with at least two observed rows: 46,940

Only validation sessions are scored. Test sessions are not used.

## Model

- Frozen checkpoint: `/Users/haozhangao/Desktop/RecSys Research/python_code_new/model_outputs/edge_mlp_full.pt`
- Model type: `edge_mlp`

## Leakage Checks

- Rank/position features in model inputs: []
- Forbidden outcome features in model inputs: []

## Interpretation

This artifact contains predicted heads for observed validation videos in observed within-session order. It does not sample unrecommended videos and does not approximate the unobserved production candidate set.

The downstream PL notebook estimates weights that rationalize observed order conditional on exposure.

## Diagnostics

```json
{
  "run_mode": "full",
  "random_seed": 20260615,
  "model_path": "/Users/haozhangao/Desktop/RecSys Research/python_code_new/model_outputs/edge_mlp_full.pt",
  "model_type": "edge_mlp",
  "split_summary": [
    {
      "split": "train",
      "sessions": 444472,
      "observations": 11139616
    },
    {
      "split": "val",
      "sessions": 55559,
      "observations": 908452
    },
    {
      "split": "test",
      "sessions": 55560,
      "observations": 479844
    }
  ],
  "observed_order_summary": {
    "run_mode": "full",
    "validation_rows": 908452,
    "validation_sessions": 55559,
    "validation_users": 6809,
    "sessions_with_at_least_2_rows": 46940,
    "rows_contributing_to_pl": 899833
  },
  "prediction_diagnostics": {
    "observed_predictions": 908452,
    "all_heads_in_unit_interval": true,
    "missing_head_predictions": 0,
    "user_cold_start_share": 0.0,
    "video_cold_start_share": 0.06433251289005913
  },
  "rank_list_checks": {
    "rank_list_rows": 908452,
    "rank_list_sessions": 55559,
    "sessions_with_K_lt_2": 8619
  },
  "rank_features_used": [],
  "forbidden_outcome_inputs_used": []
}
```
