from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "Figure5_sensor_level_values.csv"
OUTPUT = ROOT / "source_data" / "TableS13_absolute_node_gain_sensitivity.csv"
DESCRIPTORS = [
    ("absolute_bias", "Absolute residual bias"),
    ("lag1_strength", "Lag-1 dependence"),
    ("daily_strength", "Daily recurrence"),
    ("weekly_strength", "Weekly recurrence"),
    ("network_coherence", "Network coherence"),
]


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    ranks = [rankdata(values) for values in (x, y, control)]
    design = np.column_stack([np.ones(len(x)), ranks[2]])
    residual_x = ranks[0] - design @ np.linalg.lstsq(design, ranks[0], rcond=None)[0]
    residual_y = ranks[1] - design @ np.linalg.lstsq(design, ranks[1], rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def interval(
    x: np.ndarray,
    y: np.ndarray,
    control: np.ndarray | None,
    seed: int,
    repeats: int = 5000,
) -> tuple[float, float, float]:
    if control is None:
        estimate = float(spearmanr(x, y).statistic)
    else:
        estimate = partial_spearman(x, y, control)
    generator = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = generator.integers(0, len(x), len(x))
        if control is None:
            values[index] = spearmanr(x[selected], y[selected]).statistic
        else:
            values[index] = partial_spearman(
                x[selected], y[selected], control[selected]
            )
    values = values[np.isfinite(values)]
    return estimate, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    frame = pd.read_csv(DATA)
    frame["test_mae_gain_absolute"] = frame["frozen_test_mae"] - frame["cmrr_test_mae"]
    rows: list[dict] = []
    for dataset_index, dataset in enumerate(("PeMS04", "PeMS08")):
        subset = frame[frame.dataset == dataset]
        gain = subset["test_mae_gain_absolute"].to_numpy(float)
        control = subset["frozen_val_mae"].to_numpy(float)
        for descriptor_index, (key, label) in enumerate(DESCRIPTORS):
            x = subset[key].to_numpy(float)
            for association_index, association in enumerate(("Spearman", "Partial Spearman")):
                estimate, low, high = interval(
                    x,
                    gain,
                    None if association == "Spearman" else control,
                    seed=20261004 + 100 * dataset_index + 10 * descriptor_index + association_index,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "descriptor": key,
                        "descriptor_label": label,
                        "outcome": "absolute test MAE reduction",
                        "association": association,
                        "estimate": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                        "controls_for": "none" if association == "Spearman" else "frozen validation node MAE",
                        "sensors": len(subset),
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
