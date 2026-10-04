from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mask = true != 0
    error = pred[mask] - true[mask]
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "MAPE": float(100.0 * (np.abs(error) / np.maximum(np.abs(true[mask]), 1e-6)).mean()),
    }


@njit(cache=True)
def _causal_memory_core(
    pred: np.ndarray,
    true: np.ndarray,
    alpha: float,
    clip: float,
    node_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples, horizons, nodes = pred.shape
    node_memory = np.zeros((horizons, nodes), dtype=np.float32)
    global_memory = np.zeros(horizons, dtype=np.float32)
    corrected = np.empty_like(pred)
    for index in range(samples):
        # h is zero based. Its target becomes observable h+1 origins later.
        for horizon in range(horizons):
            revealed = index - (horizon + 1)
            if revealed < 0:
                continue
            valid_errors = np.empty(nodes, dtype=np.float32)
            count = 0
            for node in range(nodes):
                if true[revealed, horizon, node] != 0:
                    error = true[revealed, horizon, node] - pred[revealed, horizon, node]
                    error = min(max(error, -clip), clip)
                    node_memory[horizon, node] = (
                        (1.0 - alpha) * node_memory[horizon, node] + alpha * error
                    )
                    valid_errors[count] = error
                    count += 1
            if count:
                global_memory[horizon] = (
                    (1.0 - alpha) * global_memory[horizon]
                    + alpha * np.median(valid_errors[:count])
                )
        correction = node_weight * node_memory + (1.0 - node_weight) * global_memory[:, None]
        corrected[index] = pred[index] + correction
    return corrected, node_memory, global_memory


def causal_memory(
    pred: np.ndarray,
    true: np.ndarray,
    alpha: float,
    clip: float,
    node_weight: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    corrected, node_memory, global_memory = _causal_memory_core(
        np.asarray(pred, dtype=np.float32), np.asarray(true, dtype=np.float32),
        float(alpha), float(clip), float(node_weight)
    )
    return corrected, (node_memory, global_memory)


@njit(cache=True)
def _causal_memory_stress_core(
    pred: np.ndarray,
    true: np.ndarray,
    alpha: float,
    clip: float,
    node_weight: float,
    extra_delay: int,
    feedback_keep: float,
    seed: int,
) -> np.ndarray:
    """Chronological replay with additional label delay and telemetry loss."""
    samples, horizons, nodes = pred.shape
    node_memory = np.zeros((horizons, nodes), dtype=np.float32)
    global_memory = np.zeros(horizons, dtype=np.float32)
    corrected = np.empty_like(pred)
    modulus = 1000003
    for index in range(samples):
        for horizon in range(horizons):
            revealed = index - (horizon + 1 + extra_delay)
            if revealed < 0:
                continue
            valid_errors = np.empty(nodes, dtype=np.float32)
            count = 0
            for node in range(nodes):
                # A deterministic hash makes loss masks reproducible and nested
                # across missing-feedback levels without materializing a tensor.
                hashed = (revealed * 73856093 + horizon * 19349663 + node * 83492791 + seed) % modulus
                observed = (hashed / modulus) < feedback_keep
                if observed and true[revealed, horizon, node] != 0:
                    error = true[revealed, horizon, node] - pred[revealed, horizon, node]
                    error = min(max(error, -clip), clip)
                    node_memory[horizon, node] = (
                        (1.0 - alpha) * node_memory[horizon, node] + alpha * error
                    )
                    valid_errors[count] = error
                    count += 1
            if count:
                global_memory[horizon] = (
                    (1.0 - alpha) * global_memory[horizon]
                    + alpha * np.median(valid_errors[:count])
                )
        correction = node_weight * node_memory + (1.0 - node_weight) * global_memory[:, None]
        corrected[index] = pred[index] + correction
    return corrected


def causal_memory_stress(
    pred: np.ndarray,
    true: np.ndarray,
    alpha: float,
    clip: float,
    node_weight: float,
    extra_delay: int = 0,
    missing_fraction: float = 0.0,
    seed: int = 2026,
) -> np.ndarray:
    if extra_delay < 0:
        raise ValueError("extra_delay must be non-negative")
    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must lie in [0,1)")
    return _causal_memory_stress_core(
        np.asarray(pred, dtype=np.float32), np.asarray(true, dtype=np.float32),
        float(alpha), float(clip), float(node_weight), int(extra_delay),
        float(1.0 - missing_fraction), int(seed),
    )


@njit(cache=True)
def _causal_components_core(
    pred: np.ndarray, true: np.ndarray, alpha: float, clip: float
) -> tuple[np.ndarray, np.ndarray]:
    samples, horizons, nodes = pred.shape
    node_memory = np.zeros((horizons, nodes), dtype=np.float32)
    global_memory = np.zeros(horizons, dtype=np.float32)
    node_trace = np.zeros_like(pred)
    global_trace = np.zeros((samples, horizons), dtype=np.float32)
    for index in range(samples):
        for horizon in range(horizons):
            revealed = index - (horizon + 1)
            if revealed < 0:
                continue
            valid_errors = np.empty(nodes, dtype=np.float32)
            count = 0
            for node in range(nodes):
                if true[revealed, horizon, node] != 0:
                    error = true[revealed, horizon, node] - pred[revealed, horizon, node]
                    error = min(max(error, -clip), clip)
                    node_memory[horizon, node] = (
                        (1.0 - alpha) * node_memory[horizon, node] + alpha * error
                    )
                    valid_errors[count] = error
                    count += 1
            if count:
                global_memory[horizon] = (
                    (1.0 - alpha) * global_memory[horizon]
                    + alpha * np.median(valid_errors[:count])
                )
        node_trace[index] = node_memory
        global_trace[index] = global_memory
    return node_trace, global_trace


def causal_components(
    pred: np.ndarray, true: np.ndarray, alpha: float, clip: float
) -> tuple[np.ndarray, np.ndarray]:
    return _causal_components_core(
        np.asarray(pred, dtype=np.float32), np.asarray(true, dtype=np.float32),
        float(alpha), float(clip)
    )


def block_bootstrap(
    base: np.ndarray,
    proposed: np.ndarray,
    true: np.ndarray,
    block: int = 288,
    repeats: int = 2000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    samples = true.shape[0]
    starts = np.arange(0, samples, block)
    # Aggregate sufficient statistics once. Resampling these day-level sums is
    # exactly equivalent to repeatedly materializing the full tensors.
    stats = []
    for start in starts:
        slc = slice(start, min(start + block, samples))
        valid = true[slc] != 0
        target = true[slc][valid]
        base_error = base[slc][valid] - target
        proposed_error = proposed[slc][valid] - target
        stats.append([
            valid.sum(),
            np.abs(base_error).sum(), np.square(base_error).sum(),
            (np.abs(base_error) / np.maximum(np.abs(target), 1e-6)).sum(),
            np.abs(proposed_error).sum(), np.square(proposed_error).sum(),
            (np.abs(proposed_error) / np.maximum(np.abs(target), 1e-6)).sum(),
        ])
    stats = np.asarray(stats, dtype=np.float64)
    chosen = rng.integers(0, len(stats), size=(repeats, len(stats)))
    sampled = stats[chosen].sum(axis=1)
    count = sampled[:, 0]
    values = {
        "MAE": sampled[:, 1] / count - sampled[:, 4] / count,
        "RMSE": np.sqrt(sampled[:, 2] / count) - np.sqrt(sampled[:, 5] / count),
        "MAPE": 100.0 * (sampled[:, 3] / count - sampled[:, 6] / count),
    }
    result = {}
    for key, array in values.items():
        result[key] = {
            "mean_improvement": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
            "probability_improvement": float((array > 0).mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.prediction_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    val = dict(np.load(source / "val.npz"))
    test = dict(np.load(source / "test.npz"))
    raw_val = metrics(val["pred"], val["true"])
    def evaluate_candidate(
        alpha: float, clip: float, node_weight: float,
        node_trace: np.ndarray, global_trace: np.ndarray,
    ) -> dict[str, float]:
        corrected = val["pred"] + node_weight * node_trace + (1.0 - node_weight) * global_trace[..., None]
        score_metrics = metrics(corrected, val["true"])
        score = float(np.mean([score_metrics[key] / raw_val[key] for key in ("MAE", "RMSE", "MAPE")]))
        return {
            "score": score,
            "alpha": alpha,
            "clip": clip,
            "node_weight": float(node_weight),
            **{f"val_{key}": value for key, value in score_metrics.items()},
        }

    # Two-stage validation search: locate the memory dynamics with a coarse
    # shrinkage grid, then refine only the node/global mixing coefficient.
    candidates = []
    for alpha in (0.005, 0.01, 0.02, 0.05, 0.1):
        for clip in (5.0, 10.0, 20.0):
            node_trace, global_trace = causal_components(val["pred"], val["true"], alpha, clip)
            for node_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                candidates.append(evaluate_candidate(
                    alpha, clip, node_weight, node_trace, global_trace
                ))
    coarse = min(candidates, key=lambda item: item["score"])
    node_trace, global_trace = causal_components(
        val["pred"], val["true"], coarse["alpha"], coarse["clip"]
    )
    refined = [
        evaluate_candidate(
            coarse["alpha"], coarse["clip"], node_weight, node_trace, global_trace
        )
        for node_weight in np.arange(0.0, 1.01, 0.1)
    ]
    candidates.extend(refined)
    selected = min(refined, key=lambda item: item["score"])
    combined_pred = np.concatenate([val["pred"], test["pred"]])
    combined_true = np.concatenate([val["true"], test["true"]])
    combined_corrected, state = causal_memory(
        combined_pred,
        combined_true,
        selected["alpha"],
        selected["clip"],
        selected["node_weight"],
    )
    corrected_test = combined_corrected[len(val["pred"]):]
    base_test = metrics(test["pred"], test["true"])
    proposed_test = metrics(corrected_test, test["true"])
    per_horizon = []
    for horizon in range(test["pred"].shape[1]):
        per_horizon.append({
            "horizon": horizon + 1,
            "base": metrics(test["pred"][:, horizon], test["true"][:, horizon]),
            "proposed": metrics(corrected_test[:, horizon], test["true"][:, horizon]),
        })
    result = {
        "method": "Causal Error Memory",
        "selection_metric": "mean normalized validation MAE/RMSE/MAPE",
        "selected": selected,
        "base_test": base_test,
        "proposed_test": proposed_test,
        "relative_improvement_percent": {
            key: 100.0 * (base_test[key] - proposed_test[key]) / base_test[key]
            for key in base_test
        },
        "per_horizon": per_horizon,
        "block_bootstrap": block_bootstrap(test["pred"], corrected_test, test["true"]),
        "causality": "At origin i, horizon h memory only receives the error from origin i-(h+1).",
        "validation_samples": int(len(val["pred"])),
        "test_samples": int(len(test["pred"])),
    }
    np.savez_compressed(output / "corrected_test.npz", pred=corrected_test, true=test["true"])
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "selection_grid.json").write_text(json.dumps(sorted(candidates, key=lambda x: x["score"]), indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
