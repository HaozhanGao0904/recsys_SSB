"""Utilities for observed-order Plackett-Luce head-weight estimation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softplus
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


HEAD_COLS = ["p_complete", "p_long", "p_rewatch", "p_neg"]
WEIGHT_NAMES = ["w_complete", "w_long", "w_rewatch", "w_neg"]


@dataclass
class PLFitResult:
    weights: np.ndarray
    loss: float
    initial_loss: float
    converged: bool
    message: str
    num_iterations: int
    gradient_norm: float
    parameterization: str
    sign_constraints: bool


def compute_scores(df: pd.DataFrame, weights: np.ndarray, head_cols: list[str] | None = None) -> np.ndarray:
    """Compute linear value scores from predicted heads."""
    cols = HEAD_COLS if head_cols is None else head_cols
    return df[cols].to_numpy(dtype=np.float64) @ np.asarray(weights, dtype=np.float64)


def _weights_from_softplus_params(params: np.ndarray, sign_constraints: bool) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    if not sign_constraints:
        return params.copy(), np.ones_like(params)

    positive = softplus(params[:3])
    negative = -softplus(params[3:4])
    weights = np.r_[positive, negative]

    sigmoid = 1.0 / (1.0 + np.exp(-params))
    jac_diag = np.r_[sigmoid[:3], -sigmoid[3:4]]
    return weights, jac_diag


def prepare_session_arrays(
    df: pd.DataFrame,
    head_cols: list[str] | None = None,
    session_col: str = "session_key",
    rank_col: str = "rank_order",
) -> dict[str, object]:
    """Sort observed rows by session/rank and return arrays for PL optimization."""
    cols = HEAD_COLS if head_cols is None else head_cols
    required = [session_col, rank_col] + cols
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for PL estimation: {missing}")

    ordered = df.sort_values([session_col, rank_col, "video_id"], kind="mergesort").reset_index(drop=True)
    session_sizes = ordered.groupby(session_col, observed=True, sort=False).size().to_numpy(dtype=np.int64)
    starts = np.r_[0, np.cumsum(session_sizes)[:-1]]
    ends = np.cumsum(session_sizes)
    X = ordered[cols].to_numpy(dtype=np.float64)
    return {"df": ordered, "X": X, "starts": starts, "ends": ends, "session_sizes": session_sizes}


def pl_session_loss(scores_in_order: np.ndarray, tau: float = 1.0) -> float:
    """Rank-normalized negative PL log likelihood for one session."""
    z = np.asarray(scores_in_order, dtype=np.float64) / float(tau)
    k = len(z)
    if k < 2:
        return 0.0
    total = 0.0
    for r in range(k - 1):
        total += logsumexp(z[r:]) - z[r]
    return float(total / max(k - 1, 1))


def _pl_loss_grad_for_weights(
    weights: np.ndarray,
    arrays: dict[str, object],
    tau: float = 1.0,
    lambda_l2: float = 1e-3,
    session_weighting: str = "session",
) -> tuple[float, np.ndarray]:
    X = arrays["X"]
    starts = arrays["starts"]
    ends = arrays["ends"]

    weights = np.asarray(weights, dtype=np.float64)
    z_all = (X @ weights) / float(tau)
    grad_w = np.zeros_like(weights)
    total_loss = 0.0
    total_units = 0.0

    for start, end in zip(starts, ends):
        z = z_all[start:end]
        Xs = X[start:end]
        k = end - start
        if k < 2:
            continue

        denom = max(k - 1, 1) if session_weighting == "session" else 1.0
        # PL suffix denominators can be computed in O(K) by a reverse
        # cumulative logaddexp. The older implementation recomputed
        # logsumexp(z[r:]) for every r, which made full validation runs slow.
        suffix_logdenom = np.empty(k, dtype=np.float64)
        suffix_logdenom[-1] = z[-1]
        for idx in range(k - 2, -1, -1):
            suffix_logdenom[idx] = np.logaddexp(z[idx], suffix_logdenom[idx + 1])

        active_logdenom = suffix_logdenom[:-1]
        session_loss = float(np.sum(active_logdenom - z[:-1]))

        # For item m, gradient contribution from all choice sets r <= m is:
        # exp(z_m) * sum_{r<=m} exp(-logsumexp(z[r:])).
        # The final item appears in all K-1 choice sets but has no -z_m term.
        inv_denom_prefix = np.cumsum(np.exp(-active_logdenom))
        session_grad_z = np.empty(k, dtype=np.float64)
        session_grad_z[:-1] = np.exp(z[:-1]) * inv_denom_prefix
        session_grad_z[-1] = np.exp(z[-1]) * inv_denom_prefix[-1]
        session_grad_z[:-1] -= 1.0

        if session_weighting == "session":
            total_loss += session_loss / denom
            grad_w += (Xs.T @ (session_grad_z / denom)) / float(tau)
            total_units += 1.0
        elif session_weighting == "item":
            total_loss += session_loss
            grad_w += (Xs.T @ session_grad_z) / float(tau)
            total_units += float(k - 1)
        else:
            raise ValueError("session_weighting must be 'session' or 'item'")

    total_units = max(total_units, 1.0)
    loss = total_loss / total_units + float(lambda_l2) * float(weights @ weights)
    grad = grad_w / total_units + 2.0 * float(lambda_l2) * weights
    return float(loss), grad


def pl_total_loss(
    weights: np.ndarray,
    df: pd.DataFrame,
    tau: float = 1.0,
    lambda_l2: float = 1e-3,
    session_weighting: str = "session",
) -> float:
    """Total PL objective for a dataframe of observed validation rows."""
    arrays = prepare_session_arrays(df)
    return _pl_loss_grad_for_weights(weights, arrays, tau, lambda_l2, session_weighting)[0]


def fit_pl_weights(
    df: pd.DataFrame,
    sign_constraints: bool = True,
    tau: float = 1.0,
    lambda_l2: float = 1e-3,
    session_weighting: str = "session",
    parameterization: str = "softplus",
    start: np.ndarray | None = None,
    verbose: bool = False,
    callback_every: int = 1,
    maxiter: int = 1000,
) -> PLFitResult:
    """Fit four head weights with a PL ranking loss."""
    callback_every = max(1, int(callback_every))

    def maybe_print(message: str) -> None:
        if verbose:
            print(message, flush=True)

    label_hint = (
        "softplus constrained"
        if sign_constraints and parameterization == "softplus"
        else "bounded constrained"
        if sign_constraints
        else "unconstrained"
    )
    maybe_print(
        f"[PL {label_hint}] starting fit: rows={len(df):,} "
        f"sessions={df['session_key'].nunique():,} tau={tau} "
        f"lambda_l2={lambda_l2} session_weighting={session_weighting}"
    )
    maybe_print(f"[PL {label_hint}] preparing sorted session arrays...")
    arrays = prepare_session_arrays(df)
    maybe_print(
        f"[PL {label_hint}] prepared arrays: rows={arrays['X'].shape[0]:,} "
        f"sessions={len(arrays['starts']):,}; computing initial loss..."
    )

    if sign_constraints and parameterization == "softplus":
        if start is None:
            start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        def fun_and_grad(params):
            weights, jac_diag = _weights_from_softplus_params(params, True)
            loss, grad_w = _pl_loss_grad_for_weights(weights, arrays, tau, lambda_l2, session_weighting)
            return loss, grad_w * jac_diag

        initial_weights, _ = _weights_from_softplus_params(np.asarray(start, dtype=np.float64), True)
        initial_loss = _pl_loss_grad_for_weights(initial_weights, arrays, tau, lambda_l2, session_weighting)[0]
        maybe_print(f"[PL softplus constrained] initial_loss={initial_loss:.8f}")

        iteration = {"n": 0}

        def callback(params):
            iteration["n"] += 1
            if iteration["n"] % callback_every == 0:
                loss, grad = fun_and_grad(params)
                weights, _ = _weights_from_softplus_params(params, True)
                maybe_print(
                    "[PL softplus constrained] "
                    f"iter={iteration['n']} loss={loss:.8f} "
                    f"grad_norm={np.linalg.norm(grad):.6g} weights={weights.tolist()}"
                )

        result = minimize(
            fun=lambda a: fun_and_grad(a)[0],
            jac=lambda a: fun_and_grad(a)[1],
            x0=np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            callback=callback,
            options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
        )
        weights, jac_diag = _weights_from_softplus_params(result.x, True)
        final_grad = fun_and_grad(result.x)[1]
        maybe_print(
            "[PL softplus constrained] done "
            f"success={result.success} iterations={result.nit} final_loss={float(result.fun):.8f}"
        )
        return PLFitResult(
            weights=weights,
            loss=float(result.fun),
            initial_loss=float(initial_loss),
            converged=bool(result.success),
            message=str(result.message),
            num_iterations=int(result.nit),
            gradient_norm=float(np.linalg.norm(final_grad)),
            parameterization=parameterization,
            sign_constraints=True,
        )

    if start is None:
        start = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float64) if sign_constraints else np.zeros(4)
    bounds = [(0.0, None), (0.0, None), (0.0, None), (None, 0.0)] if sign_constraints else None
    label = "bounded constrained" if sign_constraints else "unconstrained"
    initial_loss = _pl_loss_grad_for_weights(start, arrays, tau, lambda_l2, session_weighting)[0]
    maybe_print(f"[PL {label}] initial_loss={initial_loss:.8f} start={np.asarray(start, dtype=np.float64).tolist()}")

    iteration = {"n": 0}

    def callback(weights_now):
        iteration["n"] += 1
        if iteration["n"] % callback_every == 0:
            loss, grad = _pl_loss_grad_for_weights(weights_now, arrays, tau, lambda_l2, session_weighting)
            maybe_print(
                f"[PL {label}] iter={iteration['n']} loss={loss:.8f} "
                f"grad_norm={np.linalg.norm(grad):.6g} weights={np.asarray(weights_now).tolist()}"
            )

    result = minimize(
        fun=lambda w: _pl_loss_grad_for_weights(w, arrays, tau, lambda_l2, session_weighting)[0],
        jac=lambda w: _pl_loss_grad_for_weights(w, arrays, tau, lambda_l2, session_weighting)[1],
        x0=np.asarray(start, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    final_loss, final_grad = _pl_loss_grad_for_weights(result.x, arrays, tau, lambda_l2, session_weighting)
    maybe_print(
        f"[PL {label}] done success={result.success} "
        f"iterations={result.nit} final_loss={float(final_loss):.8f}"
    )
    return PLFitResult(
        weights=result.x.astype(float),
        loss=float(final_loss),
        initial_loss=float(initial_loss),
        converged=bool(result.success),
        message=str(result.message),
        num_iterations=int(result.nit),
        gradient_norm=float(np.linalg.norm(final_grad)),
        parameterization="bounded" if sign_constraints else "unconstrained",
        sign_constraints=bool(sign_constraints),
    )


def random_pl_baseline(df: pd.DataFrame, session_col: str = "session_key") -> dict[str, float]:
    """Expected rank-normalized PL NLL for uniform random order over observed lists."""
    losses = []
    for k in df.groupby(session_col, observed=True).size().to_numpy(dtype=np.int64):
        if k >= 2:
            losses.append(float(np.log(np.arange(k, 1, -1)).sum() / (k - 1)))
    return {
        "pl_nll": float(np.mean(losses)) if losses else np.nan,
        "sessions": int(len(losses)),
    }


def score_pl_metrics(
    df: pd.DataFrame,
    score_col: str,
    tau: float = 1.0,
    session_col: str = "session_key",
    rank_col: str = "rank_order",
) -> dict[str, float]:
    """Compute rank-normalized PL NLL for an existing score column."""
    losses = []
    for _, group in df.sort_values([session_col, rank_col], kind="mergesort").groupby(session_col, observed=True, sort=False):
        if len(group) >= 2:
            losses.append(pl_session_loss(group[score_col].to_numpy(dtype=np.float64), tau=tau))
    return {
        "pl_nll": float(np.mean(losses)) if losses else np.nan,
        "sessions": int(len(losses)),
    }


def pairwise_order_metrics(
    df: pd.DataFrame,
    weights: np.ndarray | None = None,
    tau: float = 1.0,
    score_col: str | None = None,
    session_col: str = "session_key",
    rank_col: str = "rank_order",
) -> dict[str, float]:
    """Exact within-session pairwise metrics for earlier-over-later order."""
    if score_col is None:
        if weights is None:
            raise ValueError("Provide either weights or score_col")
        scores = compute_scores(df, weights)
        work = df[[session_col, rank_col]].copy()
        work["_score"] = scores
        score_col = "_score"
    else:
        work = df[[session_col, rank_col, score_col]].copy()

    margins = []
    for _, group in work.sort_values([session_col, rank_col], kind="mergesort").groupby(session_col, observed=True, sort=False):
        s = group[score_col].to_numpy(dtype=np.float64)
        k = len(s)
        if k < 2:
            continue
        for r in range(k - 1):
            margins.append(s[r] - s[(r + 1):])

    if not margins:
        return {"num_pairs": 0, "pairwise_order_accuracy": np.nan, "pairwise_order_auc": np.nan, "mean_margin": np.nan, "median_margin": np.nan}

    margins = np.concatenate(margins)
    labels = np.r_[np.ones(len(margins)), np.zeros(len(margins))]
    auc_scores = np.r_[margins / float(tau), -margins / float(tau)]
    return {
        "num_pairs": int(len(margins)),
        "pairwise_order_accuracy": float((margins > 0).mean()),
        "pairwise_order_auc": float(roc_auc_score(labels, auc_scores)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
    }


def average_session_spearman(df: pd.DataFrame, score_col: str, session_col: str = "session_key", rank_col: str = "rank_order") -> dict[str, float]:
    """Average Spearman correlation between score and earlier position within sessions."""
    values = []
    for _, group in df.groupby(session_col, observed=True, sort=False):
        if len(group) < 2 or group[score_col].nunique(dropna=False) < 2:
            continue
        corr = spearmanr(-group[rank_col].to_numpy(dtype=np.float64), group[score_col].to_numpy(dtype=np.float64)).correlation
        if np.isfinite(corr):
            values.append(float(corr))
    return {
        "sessions": int(len(values)),
        "mean_spearman_score_vs_early_position": float(np.mean(values)) if values else np.nan,
        "median_spearman_score_vs_early_position": float(np.median(values)) if values else np.nan,
    }
