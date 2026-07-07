# RecSys SSB

This repository contains the research notebooks and compact replication artifacts for **From Completion to Preference: Strategic Search Bias and Universal Debiasing**.

The project studies **Strategic Search Bias (SSB)** in short-video recommendation. The central idea is that completion, watch time, and related engagement labels are not direct measures of intrinsic video utility. They are jointly shaped by:

- the utility of the current video,
- the user's search cost,
- the user's belief about future recommendations,
- and the recommender's evolving concentration of content across categories.

The repository supports the paper's main empirical components:

1. **Structural estimation of user utility and search behavior.**
2. **Posterior expected utility (EU) labels and scalable EU prediction.**
3. **Semi-synthetic validation of the debiased EU recommender.**
4. **Causal evidence that recommendation-induced belief shocks change user selectivity.**

## Main Results

### Structural Utility and EU Prediction

The structural sample contains:

| Quantity | Value |
|---|---:|
| Users | 1,777 |
| Level-1 categories | 39 |
| Interactions | 4.33 million |

The structural model estimates heterogeneous category preferences, search costs, contextual watch-ratio effects, posterior latent states, and posterior expected utility for observed interactions.

An edge MLP predicts posterior EU out of sample with:

| Metric | Value |
|---|---:|
| Test MAE | 0.355 |
| Spearman correlation with EU | 0.963 |

This shows that the structural EU target can be scaled without re-estimating the full structural model for every future interaction.

### Semi-Synthetic Recommender Validation

The validation environment is calibrated to the structural estimates. Each user receives random-exploration recommendations, the validation estimator recovers utility/search parameters from those data, and two recommenders rank the same held-out candidate videos:

- **Baseline:** completion-centered score ranking.
- **Debiased EU recommender:** ranking by predicted posterior expected utility.

Held-out recommendation quality:

| K | Baseline utility | Debiased EU utility | Gain | Baseline regret | Debiased EU regret | NDCG baseline | NDCG EU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.340 | 2.705 | +0.365 | 3.363 | 2.998 | 0.641 | 0.677 |
| 50 | 2.279 | 2.631 | +0.352 | 2.144 | 1.792 | 0.720 | 0.760 |
| 100 | 2.249 | 2.590 | +0.341 | 1.667 | 1.326 | 0.762 | 0.803 |

At `K = 100`, the debiased EU recommender raises mean utility by 15.2%, reduces oracle regret by 20.4%, and improves NDCG from 0.762 to 0.803.

The key validation outputs are in:

```text
outputs/validation/
```

### KuaiRand Random-Exposure Transfer Check

The KuaiRand notebooks test the KuaiRec-trained transfer models on KuaiRand random-exposure rows. Because most KuaiRand exposure timestamps contain multiple videos, the reporting sample keeps only singleton random timestamps and drops clicked rows to approximate the one-video autoplay UI.

The reported comparison ranks the cleaned random rows by three scores:

- **EU:** predicted posterior expected utility.
- **Naive heads:** `pred_p_complete + pred_p_long + pred_p_rewatch - pred_p_neg`.
- **Completion only:** predicted completion probability.

Mean observed metrics among top-ranked rows:

| Top share | Metric | EU | Naive heads | Completion only |
|---:|---|---:|---:|---:|
| 10% | Follow | 0.000382 | 0.000000 | 0.000000 |
| 10% | Hate | 0.000000 | 0.000000 | 0.000382 |
| 10% | Like | 0.004589 | 0.002677 | 0.004971 |
| 10% | Profile enter | 0.004207 | 0.002677 | 0.002294 |
| 10% | Play time, ms | 2554.894 | 2755.459 | 2602.088 |
| 10% | Watch ratio | 0.088724 | 0.168943 | 0.224324 |
| 20% | Follow | 0.000382 | 0.000191 | 0.000574 |
| 20% | Hate | 0.000000 | 0.000765 | 0.000765 |
| 20% | Like | 0.004589 | 0.004398 | 0.004971 |
| 20% | Profile enter | 0.004971 | 0.003250 | 0.003633 |
| 20% | Play time, ms | 2503.642 | 2680.959 | 2505.447 |
| 20% | Watch ratio | 0.080808 | 0.134190 | 0.179096 |

The KuaiRand notebooks are in:

```text
notebooks/kuairand_random_validation/
```

### Causal Evidence for Strategic Search Bias

The causal analysis tests whether recommendation-induced changes in perceived future utility cause users to become more selective.

The treatment compares the impression induced by the real recommender with the impression induced by a leakage-free simulated recommender:

- `D_mu`: actual-minus-toy shock to expected future utility.
- `D_sigma`: actual-minus-toy shock to perceived future utility dispersion.

The outcome is the residualized change in a freely estimated, behavior-implied selectivity threshold.

Final DML estimates:

| Treatment shock | Coefficient | Clustered SE | p-value |
|---|---:|---:|---:|
| Impression mean, `D_mu` | 11.742 | 4.749 | 0.0139 |
| Impression std. dev., `D_sigma` | 18.177 | 6.617 | 0.0063 |

The preferred causal sample contains 3,918 adjacent-session transitions from 356 held-out users. Standard errors are clustered by user. The preferred sample removes threshold-boundary estimates and trims treatment outliers at the 1st/99th percentiles.

The final causal table is:

```text
outputs/causal/causal_final_dml_report_table.csv
```

## Repository Structure

```text
notebooks/               Ordered research notebooks, 01 through 11e
notebooks/kuairand_random_validation/
                         KuaiRand random-exposure transfer-validation notebooks
src/                     Small reusable helper code
docs/                    Supporting method notes for validation/head-weight estimation
data/raw_metadata/       GitHub-safe KuaiRec metadata files
data/processed_metadata/ Compact processed metadata used by the notebooks
outputs/structural/      Compact structural-estimation artifacts
outputs/causal/          Final causal-inference artifacts and report tables
outputs/validation/      Recommender validation summary tables
manifests/               File inventory and large-data notes
```

## Notebook Guide

| Stage | Notebooks | Role |
|---|---|---|
| Data and GNN setup | `01` to `03` | Prepare KuaiRec features and train the initial multi-head predictor. |
| Head validation and weights | `04a`, `04b` | Validate response heads and estimate score weights. |
| Structural model | `05`, `06` | Simulate beliefs and estimate structural utility and search parameters. |
| Expected utility and true utility | `07`, `08`, `09` | Compute posterior EU, estimate semi-synthetic true utility, and train an EU predictor. |
| Recommender validation | `10a` to `10d` | Simulate validation interactions, train baseline/debiased recommenders, and evaluate ranking quality. |
| Causal inference | `11a` to `11e` | Estimate unrestricted tau, residualize delta tau, compute debiased treatment, and run final DML regression. |
| KuaiRand transfer check | `notebooks/kuairand_random_validation/01` to `05` | Build KuaiRand random-row features, train KuaiRec transfer models, and compare EU, naive-head, and completion-only rankers on singleton/no-click random rows. |

## Key Output Files

| File | Description |
|---|---|
| `outputs/validation/validation_recsys_ranking_metrics_summary.csv` | Main recommender validation metrics. |
| `outputs/validation/validation_recsys_score_correlations.csv` | Correlations between recommender scores and calibrated true utility. |
| `outputs/causal/causal_final_dml_report_table.csv` | Main causal DML regression table. |
| `outputs/causal/causal_debiased_treatment_transitions.parquet` | Test-transition treatment table from actual-minus-toy recommendation shocks. |
| `outputs/causal/causal_delta_tau_predictions.parquet` | Residualized outcome table from the delta-tau nuisance model. |
| `outputs/structural/structural_estimates_theta_phi.npz` | Compact structural parameter estimates. |
| `outputs/structural/structural_estimation_arrays.npz` | Supporting arrays from structural estimation. |

## Data Availability

This repository includes compact metadata and final/derived artifacts needed to inspect the main results. It does **not** include the largest raw and processed data matrices because normal GitHub repositories reject files over 100MB and the full workspace contains multi-GB files.

Excluded examples:

```text
KuaiRec 2.0/data/big_matrix.csv                    ~1.0GB
KuaiRec 2.0/data/small_matrix.csv                  ~387MB
KuaiRec 2.0/data/processed/processed_data.parquet  ~1.0GB
KuaiRec 2.0/data/processed/processed_data.csv      ~5.8GB
python_code_new/outputs/structural_interactions.parquet ~175MB
```

See `manifests/DATA_MANIFEST.md` for the included and excluded data inventory.

## Environment

The notebooks were developed with Python 3.11 and common scientific Python packages. Install the approximate dependencies with:

```bash
pip install -r requirements.txt
```

Large intermediate notebooks may need substantial RAM if re-run from raw KuaiRec data.

## Reproducing Results

If the compact artifacts are present, the final report tables can be regenerated with:

```text
notebooks/10d_validation_recsys_evaluation.ipynb
notebooks/11e_causal_inference_dml_reg.ipynb
```

To rerun the full pipeline from raw interactions, restore the excluded large KuaiRec files to the paths documented in `manifests/DATA_MANIFEST.md`, then run the notebooks in order.
