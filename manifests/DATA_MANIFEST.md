# Data Manifest

This manifest separates included GitHub-safe artifacts from large files intentionally excluded from Git.

## Included Raw Metadata

These files come from the KuaiRec 2.0 data folder and are small enough for a normal GitHub repository.

```text
data/raw_metadata/item_categories.csv
data/raw_metadata/item_daily_features.csv
data/raw_metadata/kuairec_caption_category.csv
data/raw_metadata/social_network.csv
data/raw_metadata/user_features.csv
data/LICENSE
```

## Included Processed Metadata

```text
data/processed_metadata/df_model_tau.parquet
data/processed_metadata/simrec_catdist_level1.parquet
data/processed_metadata/user_cat_score_moments.parquet
data/processed_metadata/valid_user_belief.parquet
```

## Included Structural Artifacts

```text
outputs/structural/structural_estimates_theta_phi.npz
outputs/structural/structural_estimation_arrays.npz
outputs/structural/structural_estimation_bundle_metadata.json
outputs/structural/structural_sessions.parquet
```

The full row-level structural interaction table is excluded because it exceeds GitHub's normal file-size limit.

## Included Causal Artifacts

```text
outputs/causal/causal_tau_choice_sessions.parquet
outputs/causal/causal_tau_choice_transitions.parquet
outputs/causal/causal_delta_tau_predictions.parquet
outputs/causal/causal_debiased_treatment_transitions.parquet
outputs/causal/causal_final_dml_report_table.csv
outputs/causal/causal_final_dml_regression_results.csv
outputs/causal/causal_final_dml_regression_results.json
```

Small model artifacts used by the causal treatment construction are under:

```text
outputs/causal/model_artifacts/
```

## Included Validation Summaries

```text
outputs/validation/validation_recsys_ranking_metrics_summary.csv
outputs/validation/validation_recsys_score_correlations.csv
outputs/validation/validation_recsys_evaluation_metrics.json
outputs/validation/validation_baseline_metrics.json
outputs/validation/validation_debiased_em06_metrics.json
```

Large validation candidate and target tables are excluded.

## Excluded Large Files

These files remain in the local research workspace but are intentionally not copied into this repository.

| Local file | Approximate size | Reason |
|---|---:|---|
| `KuaiRec 2.0/data/big_matrix.csv` | 1.0GB | Raw interaction matrix; exceeds GitHub limit. |
| `KuaiRec 2.0/data/small_matrix.csv` | 387MB | Raw interaction matrix; exceeds GitHub limit. |
| `KuaiRec 2.0/data/processed/processed_data.parquet` | 1.0GB | Full processed row-level data; exceeds GitHub limit. |
| `KuaiRec 2.0/data/processed/processed_data.csv` | 5.8GB | Full processed row-level data; exceeds GitHub limit. |
| `KuaiRec 2.0/data/processed/gnn_data.pt` | 6.7GB | Full GNN tensor artifact; exceeds GitHub limit. |
| `KuaiRec 2.0/data/processed/precomputed_heads.pkl` | 13GB | Large cached model output; exceeds GitHub limit. |
| `python_code_new/outputs/structural_interactions.parquet` | 175MB | Row-level structural sample; exceeds GitHub limit. |
| `python_code_new/outputs/semi_synth_true_utility.parquet` | 382MB | Large semi-synthetic utility table; exceeds GitHub limit. |
| `python_code_new/outputs/validation_simulation/validation_exploration_interactions.parquet` | 601MB | Large validation simulation table; exceeds GitHub limit. |

For a full rerun, restore these files from the local workspace or an external data store using the original relative paths.
