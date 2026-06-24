# Validation Candidate Generation and Head Prediction

## Data

- Run mode: full
- Full validation sessions: 55,559
- Full validation observations: 908,452
- Active validation sessions used in this run: 55,559
- Positive session-videos: 755,654

Only validation sessions are target sessions. Test sessions are not used as positive target sessions.

## Negative Sampling

For each validation positive, the notebook sampled up to `M=5` negatives.

- negative set definition: `user_never_recommended_full_panel`
- upload-time availability restriction: `False`

Under the default `user_never_recommended_full_panel` definition, a sampled negative must be outside the user's full-panel recommended set. Therefore later same-user recommendations cannot be sampled as negatives. This intentionally uses full-panel recommendation information for behavioral cloning.

## Feature Construction

Copied from pre-session/session state:

- user static features
- session/context features
- general user-history features

Merged from candidate video metadata:

- item static features
- item category hierarchy
- author and tag metadata

Recomputed for each candidate:

- `i_age_since_upload_days`
- `hist_cat_ema_complete`
- `hist_author_recency_days`
- `hist_last_complete_author`
- `hist_has_author_history`

Forbidden realized outcomes were not used as model input features.

## Diagnostics

```json
{
  "run_mode": "full",
  "random_seed": 20260615,
  "m_negatives": 5,
  "negative_exclusion_policy": "user_never_recommended_full_panel",
  "upload_availability_restricted": false,
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
  "candidate_sample_summary": {
    "run_mode": "full",
    "validation_rows": 908452,
    "validation_sessions": 55559,
    "validation_users": 6809,
    "positive_session_videos": 755654
  },
  "pair_sampling_diagnostics": {
    "validation_sessions_used": 55559,
    "positive_videos": 755654,
    "pairs": 3778270,
    "average_negatives_per_positive": 5.0,
    "negative_shortfall_sessions": 0,
    "future_seen_false_negative_count": 0,
    "old_policy_ever_seen_negative_count": 0
  },
  "feature_diagnostics": {
    "candidate_rows": 4462598,
    "positive_candidates": 755654,
    "negative_candidates": 3706944,
    "unique_candidate_session_videos": 4462598,
    "missing_required_feature_cells": 8324411,
    "forbidden_columns_in_candidate_features": []
  },
  "prediction_diagnostics": {
    "candidate_predictions": 4462598,
    "all_heads_in_unit_interval": true,
    "missing_head_predictions": 0,
    "user_cold_start_share": 0.0,
    "video_cold_start_share": 0.11768324191423919
  },
  "pair_delta_diagnostics": {
    "pair_head_delta_rows": 3778270,
    "pairs_with_plus_minus_predictions": 3778270,
    "delta_missing_cells": 0,
    "negative_prior_seen_by_user_sum": 0,
    "negative_future_seen_by_user_sum": 0,
    "negative_ever_seen_by_user_sum": 0
  },
  "rank_features_used": [],
  "forbidden_outcome_inputs_used": []
}
```
