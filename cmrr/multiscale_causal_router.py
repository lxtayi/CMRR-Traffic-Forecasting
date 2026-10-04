from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_error_memory import block_bootstrap, causal_memory, metrics


# Phi^{-1}(3/4): median(|Z|) for Z ~ N(0,1).
NORMAL_ABSOLUTE_MEDIAN = 0.6744897501960817


def seasonal_features(pred: np.ndarray, true: np.ndarray, lags: tuple[int, ...], clip: float) -> np.ndarray:
    """Build causal node/global residual features with shape [T,H,N,2*L]."""
    features = np.zeros((*pred.shape, 2 * len(lags)), dtype=np.float32)
    for j, lag in enumerate(lags):
        if lag >= len(pred):
            continue
        # PeMS uses zero targets as invalid observations.  Keep the historical
        # residual policy consistent with the evaluation mask and the online
        # EWMA memory: an invalid target contributes neither a node residual nor
        # to the network-wide median.
        valid = true[:-lag] != 0
        residual = np.where(
            valid, np.clip(true[:-lag] - pred[:-lag], -clip, clip), 0.0
        ).astype(np.float32)
        global_residual = np.nanmedian(
            np.where(valid, residual, np.nan), axis=2, keepdims=True
        )
        global_residual = np.nan_to_num(global_residual, nan=0.0).astype(np.float32)
        features[lag:, ..., 2 * j] = residual
        features[lag:, ..., 2 * j + 1] = global_residual
    return features


def robust_coefficients(
    features: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    ridge: float,
    magnitude: np.ndarray,
    percentage_power: float,
    huber_delta: float = 1.5,
    iterations: int = 6,
) -> np.ndarray:
    """Small per-horizon robust ridge router; no node-specific fitted weights."""
    horizons = target.shape[1]
    coefficients = np.zeros((horizons, features.shape[-1]), dtype=np.float64)
    for horizon in range(horizons):
        mask = valid[:, horizon].reshape(-1)
        x = features[:, horizon].reshape(-1, features.shape[-1])[mask].astype(np.float64)
        y = target[:, horizon].reshape(-1)[mask].astype(np.float64)
        mag = magnitude[:, horizon].reshape(-1)[mask].astype(np.float64)
        # Deterministic thinning controls memory while preserving chronology.
        if len(y) > 400_000:
            keep = np.linspace(0, len(y) - 1, 400_000, dtype=np.int64)
            x, y, mag = x[keep], y[keep], mag[keep]
        coef = np.zeros(x.shape[1], dtype=np.float64)
        eye = np.eye(x.shape[1], dtype=np.float64)
        for _ in range(iterations):
            residual = y - x @ coef
            scale = max(float(np.median(np.abs(residual))) / NORMAL_ABSOLUTE_MEDIAN, 1e-3)
            threshold = huber_delta * scale
            robust_weight = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-6))
            percentage_weight = np.power(np.maximum(mag, 5.0), -percentage_power)
            weight = robust_weight * percentage_weight
            xw = x * np.sqrt(weight[:, None])
            yw = y * np.sqrt(weight)
            coef = np.linalg.solve(xw.T @ xw + ridge * eye, xw.T @ yw)
        coefficients[horizon] = np.clip(coef, -1.0, 1.0)
    return coefficients


def route(recent: np.ndarray, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    correction = np.einsum("thnf,hf->thn", features, coefficients, optimize=True)
    return recent + correction.astype(np.float32)


def purge_fitting_boundary(valid: np.ndarray, boundary: int) -> np.ndarray:
    """Remove fitting pairs whose targets fall in the selection suffix.

    Horizon indices are zero based in the arrays, so a pair at origin ``t``
    and array horizon ``h`` has target time ``t + h + 1``. The selection
    suffix begins at ``boundary``; therefore a fitting pair is retained only
    when ``t + h + 1 < boundary``.
    """
    if boundary <= 0 or boundary > valid.shape[0]:
        raise ValueError("boundary must lie within the validation sequence")
    fitting_valid = valid[:boundary].copy()
    origins = np.arange(boundary, dtype=np.int64)[:, None, None]
    horizons = np.arange(1, valid.shape[1] + 1, dtype=np.int64)[None, :, None]
    fitting_valid &= origins + horizons < boundary
    return fitting_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--recent-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lags", default="12,288,2016")
    args = parser.parse_args()
    source = Path(args.prediction_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    val = dict(np.load(source / "val.npz"))
    test = dict(np.load(source / "test.npz"))
    recent_result = json.loads(Path(args.recent_result).read_text(encoding="utf-8"))
    # A CMRR result can itself serve as the source of the selected recent
    # memory configuration when reproducing or revising an experiment.
    config = recent_result.get("selected", recent_result.get("recent_memory_config"))
    if config is None:
        raise KeyError("recent result must contain 'selected' or 'recent_memory_config'")
    lags = tuple(int(value) for value in args.lags.split(",") if value.strip())
    if not lags:
        raise ValueError("At least one causal lag is required")

    val_recent, _ = causal_memory(
        val["pred"], val["true"], config["alpha"], config["clip"], config["node_weight"]
    )
    val_features = seasonal_features(val["pred"], val["true"], lags, clip=40.0)
    valid = val["true"] != 0
    target = val["true"] - val_recent

    # Select regularization on the chronologically last validation third.
    # Purge horizon-dependent boundary pairs whose targets fall in that suffix.
    boundary = int(len(val["pred"]) * 2 / 3)
    fitting_valid = purge_fitting_boundary(valid, boundary)
    candidates = []
    for ridge in (100.0, 1000.0, 10000.0):
        for percentage_power in (0.0, 0.5, 1.0):
            coef = robust_coefficients(
                val_features[:boundary], target[:boundary], fitting_valid, ridge=ridge,
                magnitude=np.abs(val["true"][:boundary]), percentage_power=percentage_power,
            )
            prediction = route(val_recent[boundary:], val_features[boundary:], coef)
            score_metrics = metrics(prediction, val["true"][boundary:])
            base_metrics = metrics(val["pred"][boundary:], val["true"][boundary:])
            score = float(np.mean([score_metrics[k] / base_metrics[k] for k in ("MAE", "RMSE", "MAPE")]))
            candidates.append({"ridge": ridge, "percentage_power": percentage_power, "score": score, **score_metrics})
    selected = min(candidates, key=lambda item: item["score"])
    coefficients = robust_coefficients(
        val_features, target, valid, ridge=selected["ridge"],
        magnitude=np.abs(val["true"]), percentage_power=selected["percentage_power"],
    )

    combined_pred = np.concatenate([val["pred"], test["pred"]])
    combined_true = np.concatenate([val["true"], test["true"]])
    combined_recent, _ = causal_memory(
        combined_pred, combined_true, config["alpha"], config["clip"], config["node_weight"]
    )
    combined_features = seasonal_features(combined_pred, combined_true, lags, clip=40.0)
    proposed = route(combined_recent, combined_features, coefficients)[len(val["pred"]):]
    recent_test = combined_recent[len(val["pred"]):]
    base = metrics(test["pred"], test["true"])
    recent = metrics(recent_test, test["true"])
    routed = metrics(proposed, test["true"])
    per_horizon = []
    for horizon in range(test["pred"].shape[1]):
        per_horizon.append({
            "horizon": horizon + 1,
            "base": metrics(test["pred"][:, horizon], test["true"][:, horizon]),
            "recent_memory": metrics(recent_test[:, horizon], test["true"][:, horizon]),
            "proposed": metrics(proposed[:, horizon], test["true"][:, horizon]),
        })
    result = {
        "method": "Causal Multi-Scale Residual Router (CMRR)",
        "lags": list(lags),
        "recent_memory_config": config,
        "selection": selected,
        "validation_selection_protocol": {
            "boundary_origin": boundary,
            "target_overlap_purged": True,
            "purged_origins_at_horizon_h": "h for one-based horizon h",
        },
        "validation_candidates": candidates,
        "coefficients_by_horizon": coefficients.tolist(),
        "base_test": base,
        "recent_memory_test": recent,
        "proposed_test": routed,
        "per_horizon": per_horizon,
        "relative_improvement_vs_base_percent": {
            k: 100.0 * (base[k] - routed[k]) / base[k] for k in base
        },
        "block_bootstrap_vs_base": block_bootstrap(test["pred"], proposed, test["true"]),
        "block_bootstrap_vs_recent": block_bootstrap(recent_test, proposed, test["true"]),
        "trainable_parameters": int(coefficients.size),
        "causality": "Every residual feature is retrieved from an already observed origin; router weights use validation only.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output / "corrected_test.npz", pred=proposed, true=test["true"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
