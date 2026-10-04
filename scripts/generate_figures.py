from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "outputs" / "figures"

P04 = "#365C8D"
P08 = "#2A9D8F"
MAE = "#365C8D"
RMSE = "#2A9D8F"
MAPE = "#D66A5E"
INK = "#20242A"
GREY = "#7A7F86"
MID_GREY = "#A9ADB2"
LIGHT = "#E8EAED"
PALE = "#F3F4F5"
GOLD = "#D8A13B"
METRIC_COLOR = {"MAE": MAE, "RMSE": RMSE, "MAPE": MAPE}
METRIC_MARKER = {"MAE": "o", "RMSE": "s", "MAPE": "^"}
METRIC_LINESTYLE = {"MAE": "-", "RMSE": ":", "MAPE": "--"}
OBSERVED = "#5B6573"
FROZEN = "#B87457"
TOTAL = "#8A5A91"
PREDICTION_RED = "#D62728"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 300},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel(ax: plt.Axes, value: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, value, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def clean(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = False) -> None:
    if xgrid:
        ax.grid(axis="x", color=LIGHT, lw=0.45, zorder=0)
    if ygrid:
        ax.grid(axis="y", color=LIGHT, lw=0.45, zorder=0)
    ax.tick_params(length=2.5, color=GREY)


def annotate_metric_values(ax: plt.Axes, x: np.ndarray, values: np.ndarray,
                           metric: str, *, fontsize: float = 5.2) -> None:
    """Label compact metric trajectories without hiding overlapping series."""
    offset = {"MAE": 5.5, "RMSE": -8.5, "MAPE": -8.5}[metric]
    va = "bottom" if offset > 0 else "top"
    for x_i, value in zip(x, values):
        ax.annotate(
            f"{value:.2f}",
            (x_i, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            color=METRIC_COLOR[metric],
            fontsize=fontsize,
            fontweight="semibold",
            clip_on=False,
        )


def light_marker_fill(color: str, white_fraction: float = 0.58) -> tuple[float, float, float]:
    """Return a pale solid fill while preserving a darker same-hue border."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * white_fraction)


def figure3() -> None:
    horizon = pd.read_csv(DATA / "Figure3_horizon_paired_daily_block_CI.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.35), sharex=True)
    for row, dataset in enumerate(("PeMS04", "PeMS08")):
        for col, metric in enumerate(("MAE", "RMSE", "MAPE")):
            ax = axes[row, col]
            d = horizon[(horizon.dataset == dataset) & (horizon.metric == metric)].sort_values("horizon")
            x = d.horizon.to_numpy(float)
            y = d.improvement_percent.to_numpy(float)
            lo = d.ci95_low_percent.to_numpy(float)
            hi = d.ci95_high_percent.to_numpy(float)
            color = METRIC_COLOR[metric]
            ax.fill_between(x, lo, hi, color=color, alpha=0.10, lw=0)
            ax.plot(x, y, color=color, marker=METRIC_MARKER[metric], ms=2.8, lw=1.15,
                    mfc="white", mew=0.7)
            ax.axhline(0, color=GREY, lw=0.75)
            ax.set_xticks([1, 3, 6, 9, 12])
            if row == 0:
                ax.set_title(metric, loc="left", fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{dataset}\nImprovement (%)")
            if row == 1:
                ax.set_xlabel("Forecast horizon (5-min steps)")
            clean(ax, ygrid=True)
    for value, ax in zip("abcdef", axes.ravel()):
        panel(ax, value, -0.20 if value in "ad" else -0.15)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.12, hspace=0.34, wspace=0.32)
    save(fig, "Fig3_horizon_improvements_with_ci")


def figure4() -> None:
    structure = pd.read_csv(DATA / "residual_structure_before_after.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.35))
    horizon_styles = ((1, P04, "-"), (6, P08, "--"), (12, MAPE, ":"))
    remaining_color = P08
    for row, dataset in enumerate(("PeMS04", "PeMS08")):
        acf = pd.read_csv(DATA / f"Figure4_{dataset}_residual_acf.csv")
        for horizon, color, linestyle in horizon_styles:
            d = acf[(acf.horizon == horizon) & (acf.lag <= 2016)]
            axes[row, 0].plot(d.lag, d.acf, color=color, ls=linestyle, lw=0.9, label=f"h={horizon}")
        axes[row, 0].axvline(288, color=MID_GREY, lw=0.55, ls="--")
        axes[row, 0].axvline(2016, color=MID_GREY, lw=0.55, ls="--")

        nodes = pd.read_csv(DATA / f"Figure4_{dataset}_residual_nodes.csv")
        axes[row, 1].hist(nodes.mean_residual, bins=28, color=P04, alpha=0.82,
                          edgecolor="white", lw=0.25)
        axes[row, 1].axvline(0, color=GREY, lw=0.7)

        time = pd.read_csv(DATA / f"Figure4_{dataset}_residual_time.csv")
        d = time[time.horizon == 1]
        axes[row, 2].plot(d.origin / 288.0, d.network_median_residual, color=P08, lw=0.75)
        axes[row, 2].axhline(0, color=GREY, lw=0.7)

        d = structure[structure.dataset == dataset]
        summaries = []
        for label, measure in (("Lag 1", "Lag-1 ACF"), ("Daily", "Daily ACF"), ("Weekly", "Weekly ACF")):
            frozen = d[(d.method == "Frozen") & (d.measure == measure)].value.abs().mean()
            cmrr = d[(d.method == "CMRR") & (d.measure == measure)].value.abs().mean()
            summaries.append((label, 100.0 * cmrr / frozen))
        for label, measure in (("Node\nbias", "Mean absolute node bias"),
                               ("Network\nvariability", "Network-median residual variability")):
            frozen = float(d[(d.method == "Frozen") & (d.measure == measure)].value.iloc[0])
            cmrr = float(d[(d.method == "CMRR") & (d.measure == measure)].value.iloc[0])
            summaries.append((label, 100.0 * cmrr / frozen))
        labels, ratios = zip(*summaries)
        y = np.arange(len(labels))
        axes[row, 3].barh(y, ratios, color=remaining_color, alpha=0.88, height=0.58)
        axes[row, 3].axvline(100, color=GREY, lw=0.75, ls="--")
        axes[row, 3].set_yticks(y, labels)
        axes[row, 3].invert_yaxis()
        axes[row, 3].set_xlim(0, 120)

        axes[row, 0].set_ylabel(f"{dataset}\nACF")
        axes[row, 1].set_ylabel("Sensors")
        axes[row, 2].set_ylabel("Median residual")
        axes[row, 3].set_xlabel("CMRR / frozen (%)")
        if row == 0:
            axes[row, 0].set_title("Autocorrelation", loc="left", fontweight="bold")
            axes[row, 1].set_title("Node-wise bias", loc="left", fontweight="bold")
            axes[row, 2].set_title("Network shift", loc="left", fontweight="bold")
            axes[row, 3].set_title("Structure remaining", loc="left", fontweight="bold")
            axes[row, 0].legend(frameon=False, ncols=3, loc="upper right")
        if row == 1:
            axes[row, 0].set_xlabel("Lag (5-min origins)")
            axes[row, 1].set_xlabel("Mean validation residual")
            axes[row, 2].set_xlabel("Validation day")
        for ax in axes[row]:
            clean(ax, ygrid=True)
    for value, ax in zip("abcdefgh", axes.ravel()):
        panel(ax, value, -0.22 if value in "ae" else -0.18)
    fig.text(0.39, 0.985, "Validation residual structure", ha="center", va="top",
             fontsize=8.3, fontweight="bold", color=INK)
    fig.text(0.865, 0.985, "Test residual attenuation", ha="center", va="top",
             fontsize=8.3, fontweight="bold", color=INK)
    fig.add_artist(Line2D([0.095, 0.68], [0.957, 0.957], transform=fig.transFigure, color=LIGHT, lw=1.0))
    fig.add_artist(Line2D([0.735, 0.985], [0.957, 0.957], transform=fig.transFigure, color=LIGHT, lw=1.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.12, hspace=0.38, wspace=0.48)
    save(fig, "Fig4_residual_structure")


def figure6() -> None:
    objective = pd.read_csv(DATA / "router_objective_two_factor_ablation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45))
    methods = ["Ordinary ridge", "Huber only", "Magnitude only", "Huber + magnitude"]
    short = ["w/o both", "w/o magnitude", "w/o Huber", "Full CMRR"]
    for col, (dataset, color) in enumerate((("PeMS04", P04), ("PeMS08", P08))):
        ax = axes[0, col]
        d = objective[objective.dataset == dataset]
        x = np.arange(len(methods))
        for metric in ("MAE", "RMSE", "MAPE"):
            values = np.asarray([
                float(d[(d.method == method) & (d.metric == metric)].improvement_percent.iloc[0])
                for method in methods
            ])
            ax.plot(x, values, color=METRIC_COLOR[metric], marker=METRIC_MARKER[metric],
                    ls=METRIC_LINESTYLE[metric], ms=3.6, lw=1.15,
                    mfc=light_marker_fill(METRIC_COLOR[metric]),
                    mec=METRIC_COLOR[metric], mew=0.85, label=metric)
            annotate_metric_values(ax, x, values, metric)
        ax.axhline(0, color=GREY, lw=0.75)
        ax.margins(y=0.20)
        ax.set_xticks(x, short)
        ax.set_ylabel("Improvement over frozen (%)")
        ax.set_title(f"{dataset}: routing objective", loc="left", fontweight="bold")
        clean(ax, ygrid=True)

        ax = axes[1, col]
        volume = pd.read_csv(DATA / f"Figure6_{dataset}_volume_bins.csv")
        ordinary = volume[volume.method == "Ordinary ridge"].set_index("volume_bin")
        cmrr = volume[volume.method == "CMRR"].set_index("volume_bin")
        bins = ["low", "middle", "high"]
        x = np.arange(3)
        for metric in ("MAE", "MAPE"):
            values = np.asarray(
                100.0 * (ordinary.loc[bins, metric] - cmrr.loc[bins, metric]) / ordinary.loc[bins, metric]
            )
            ax.plot(x, values, color=METRIC_COLOR[metric], marker=METRIC_MARKER[metric],
                    ls=METRIC_LINESTYLE[metric], ms=3.8, lw=1.15,
                    mfc=light_marker_fill(METRIC_COLOR[metric]),
                    mec=METRIC_COLOR[metric], mew=0.85, label=metric)
            annotate_metric_values(ax, x, values, metric)
        ax.axhline(0, color=GREY, lw=0.75)
        ax.margins(y=0.22)
        ax.set_xticks(x, ["Low", "Middle", "High"])
        ax.set_xlabel("Observed traffic-flow tertile")
        ax.set_ylabel("CMRR vs ordinary ridge (%)")
        ax.set_title(f"{dataset}: flow-volume behavior", loc="left", fontweight="bold")
        clean(ax, ygrid=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center",
               bbox_to_anchor=(0.54, 0.995), columnspacing=1.5, handlelength=2.1)
    for value, ax in zip("abcd", axes.ravel()):
        panel(ax, value)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.87, bottom=0.12, hspace=0.45, wspace=0.30)
    save(fig, "Fig6_routing_objective_and_flow_behavior")


def figure7() -> None:
    example = pd.read_csv(DATA / "Figure7_correction_decomposition.csv")
    selection = pd.read_csv(DATA / "Figure7_example_selection.csv").set_index("dataset")

    # The top row gives the conventional real-versus-prediction comparison.
    # Dedicated legend rows keep every legend outside the plotted data.
    fig = plt.figure(figsize=(7.2, 6.15))
    grid = fig.add_gridspec(
        8,
        2,
        height_ratios=[2.15, 0.11, 0.17, 0.95, 0.17, 0.95, 0.11, 0.17],
        left=0.095,
        right=0.985,
        top=0.96,
        bottom=0.045,
        hspace=0.28,
        wspace=0.25,
    )
    prediction_axes = [fig.add_subplot(grid[0, col]) for col in range(2)]
    prediction_xlabel_ax = fig.add_subplot(grid[1, :])
    prediction_legend_ax = fig.add_subplot(grid[2, :])
    residual_axes = [fig.add_subplot(grid[3, col]) for col in range(2)]
    residual_legend_ax = fig.add_subplot(grid[4, :])
    correction_axes = [fig.add_subplot(grid[5, col]) for col in range(2)]
    correction_xlabel_ax = fig.add_subplot(grid[6, :])
    correction_legend_ax = fig.add_subplot(grid[7, :])
    for helper_ax in (
        prediction_xlabel_ax,
        prediction_legend_ax,
        residual_legend_ax,
        correction_xlabel_ax,
        correction_legend_ax,
    ):
        helper_ax.axis("off")

    prediction_handles = prediction_labels = None
    residual_handles = residual_labels = None
    correction_handles = correction_labels = None

    for col, (dataset, color) in enumerate((("PeMS04", P04), ("PeMS08", P08))):
        d = example[example.dataset == dataset].sort_values("origin").reset_index(drop=True)
        x = d.origin.to_numpy(float) * 5.0 / 60.0
        sensor = int(selection.loc[dataset, "sensor_index"])
        valid_target = d.observed.to_numpy(float) != 0.0

        # The public PeMS protocol treats zero-valued targets as masked.  Break
        # target-dependent traces at these origins rather than presenting the
        # benchmark null sentinel as physical zero flow.  Shading also makes
        # clear that corrections are still produced from the existing causal
        # state even though no target/residual is revealed at those origins.
        masked_spans = []
        masked = np.flatnonzero(~valid_target)
        if masked.size:
            start = previous = int(masked[0])
            for index in masked[1:]:
                index = int(index)
                if index != previous + 1:
                    masked_spans.append((start, previous))
                    start = index
                previous = index
            masked_spans.append((start, previous))

        ax = prediction_axes[col]
        observed_plot = d.observed.where(valid_target, np.nan)
        cmrr_plot = d.cmrr.where(valid_target, np.nan)
        ax.plot(x, observed_plot, color=P04, lw=1.65, label="real", zorder=2)
        ax.plot(
            x,
            cmrr_plot,
            color=PREDICTION_RED,
            lw=1.35,
            marker="o",
            markevery=24,
            ms=2.6,
            mfc="white",
            mec=PREDICTION_RED,
            mew=0.75,
            label="predict_CMRR",
            zorder=3,
        )
        ax.set_title(f"{dataset} | validation-representative sensor {sensor}", loc="left", fontweight="bold")
        ax.set_ylabel("Traffic flow")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["0", "6", "12", "18", "24"])
        clean(ax, ygrid=True)

        rax = residual_axes[col]
        backbone_residual_plot = d.backbone_residual.where(valid_target, np.nan)
        cmrr_residual_plot = d.cmrr_residual.where(valid_target, np.nan)
        rax.plot(x, backbone_residual_plot, color=FROZEN, lw=0.9,
                 ls=(0, (4.0, 2.1)), label="Frozen residual")
        rax.plot(x, cmrr_residual_plot, color=color, lw=1.1, label="CMRR residual")
        rax.axhline(0, color=MID_GREY, lw=0.65)
        rax.set_ylabel("Residual")
        rax.set_xlim(0, 24)
        rax.set_xticks([0, 6, 12, 18, 24])
        rax.tick_params(labelbottom=False)
        clean(rax, ygrid=True)

        cax = correction_axes[col]
        cax.plot(x, d.recent_component, color=P04, lw=0.70, label="Recent")
        cax.plot(x, d.daily_component, color=P08, lw=0.70, label="Daily")
        cax.plot(x, d.weekly_component, color=MAPE, lw=0.70, label="Weekly")
        cax.plot(x, d.total_correction, color=TOTAL, lw=1.35, label="Total")
        cax.axhline(0, color=MID_GREY, lw=0.65)
        cax.set_ylabel("Correction")
        cax.set_xlim(0, 24)
        cax.set_xticks([0, 6, 12, 18, 24])
        cax.set_xticklabels(["0", "6", "12", "18", "24"])
        clean(cax, ygrid=True)

        if masked_spans:
            half_step = 2.5 / 60.0
            for start, end in masked_spans:
                left = max(0.0, x[start] - half_step)
                right = min(24.0, x[end] + half_step)
                for target_ax in (ax, rax, cax):
                    target_ax.axvspan(
                        left,
                        right,
                        color=LIGHT,
                        alpha=0.42,
                        lw=0,
                        zorder=0,
                    )
                ax.text(
                    0.5 * (left + right),
                    0.055,
                    "masked targets",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    color=MID_GREY,
                )

        if col == 0:
            prediction_handles, prediction_labels = ax.get_legend_handles_labels()
            residual_handles, residual_labels = rax.get_legend_handles_labels()
            correction_handles, correction_labels = cax.get_legend_handles_labels()

    prediction_xlabel_ax.text(
        0.5, 0.5, "Hour of test day", ha="center", va="center", fontsize=7.6
    )
    prediction_legend_ax.legend(
        prediction_handles,
        prediction_labels,
        frameon=False,
        ncols=2,
        loc="center",
        columnspacing=1.8,
        handlelength=2.3,
    )
    residual_legend_ax.legend(
        residual_handles,
        residual_labels,
        frameon=False,
        ncols=2,
        loc="center",
        columnspacing=1.7,
        handlelength=2.2,
    )
    correction_xlabel_ax.text(
        0.5,
        0.5,
        "Hour of test day",
        ha="center",
        va="center",
        fontsize=7.6,
    )
    correction_legend_ax.legend(
        correction_handles,
        correction_labels,
        frameon=False,
        ncols=4,
        loc="center",
        columnspacing=1.3,
        handlelength=2.0,
    )

    for value, ax in zip("ab", prediction_axes):
        panel(ax, value, -0.13)
    for value, ax in zip("cd", residual_axes):
        panel(ax, value, -0.13)
    for value, ax in zip("ef", correction_axes):
        panel(ax, value, -0.13)
    save(fig, "Fig7_prediction_residual_correction")

def _frozen(dataset: str) -> pd.Series:
    frame = pd.read_csv(DATA / f"Figure6_{dataset}_relative_metrics.csv")
    return frame[frame.method == "Frozen backbone"].iloc[0]


def figure8() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45))
    for col, dataset in enumerate(("PeMS04", "PeMS08")):
        base = _frozen(dataset)
        stress = pd.read_csv(DATA / f"Figure8_{dataset}_feedback_stress.csv")
        delay = stress[stress.stress == "extra_delay"].sort_values("level")
        ax = axes[0, col]
        delay_x = delay.level.to_numpy(float) * 5.0
        label_delay = np.isin(delay_x, [0.0, 30.0, 60.0, 120.0])
        for metric in ("MAE", "RMSE", "MAPE"):
            gain = 100.0 * (float(base[metric]) - delay[metric].to_numpy(float)) / float(base[metric])
            ax.plot(delay_x, gain, color=METRIC_COLOR[metric],
                    marker=METRIC_MARKER[metric], ls=METRIC_LINESTYLE[metric],
                    ms=3.4, lw=1.1, mfc=light_marker_fill(METRIC_COLOR[metric]),
                    mec=METRIC_COLOR[metric], mew=0.85, label=metric)
            annotate_metric_values(ax, delay_x[label_delay], gain[label_delay], metric, fontsize=4.9)
        ax.axhline(0, color=GREY, lw=0.75)
        ax.margins(y=0.20)
        ax.set_title(f"{dataset}: delayed feedback", loc="left", fontweight="bold")
        ax.set_xlabel("Additional feedback delay (min)")
        ax.set_ylabel("Improvement over frozen (%)")
        clean(ax, ygrid=True)

        ax = axes[1, col]
        missing = pd.read_csv(DATA / f"{dataset}_missing_feedback_masks.csv")
        individual = missing[missing.summary == "individual"]
        x = np.array([0.0, 10.0, 30.0, 50.0])
        primary = stress[(stress.stress == "extra_delay") & (stress.level == 0)].iloc[0]
        for metric in ("MAE", "RMSE", "MAPE"):
            center = [100.0 * (float(base[metric]) - float(primary[metric])) / float(base[metric])]
            lower = [center[0]]
            upper = [center[0]]
            for fraction in (0.1, 0.3, 0.5):
                values = 100.0 * (float(base[metric]) - individual[individual.missing_fraction == fraction][metric].to_numpy(float)) / float(base[metric])
                center.append(float(values.mean()))
                lower.append(float(values.min()))
                upper.append(float(values.max()))
            center = np.asarray(center)
            yerr = np.vstack([center - np.asarray(lower), np.asarray(upper) - center])
            ax.errorbar(x, center, yerr=yerr, color=METRIC_COLOR[metric], marker=METRIC_MARKER[metric],
                        ls=METRIC_LINESTYLE[metric], ms=3.4, lw=1.05,
                        mfc=light_marker_fill(METRIC_COLOR[metric]),
                        mec=METRIC_COLOR[metric], mew=0.85, capsize=2.0, label=metric)
            annotate_metric_values(ax, x, center, metric, fontsize=4.9)
        ax.axhline(0, color=GREY, lw=0.75)
        ax.margins(y=0.20)
        ax.set_title(f"{dataset}: missing matured feedback", loc="left", fontweight="bold")
        ax.set_xlabel("Missing feedback (%)")
        ax.set_ylabel("Improvement over frozen (%)")
        clean(ax, ygrid=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center",
               bbox_to_anchor=(0.54, 0.995), columnspacing=1.5, handlelength=2.1)
    for value, ax in zip("abcd", axes.ravel()):
        panel(ax, value)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.87, bottom=0.12, hspace=0.45, wspace=0.30)
    save(fig, "Fig8_feedback_stress")


def figure9() -> None:
    transfers = pd.read_csv(DATA / "Figure9_official_STID_directed_transfers.csv")
    datasets = ["PeMS03", "PeMS04", "PeMS07", "PeMS08"]
    matrices = []
    for metric in ("MAE", "RMSE", "MAPE"):
        matrix = np.full((4, 4), np.nan)
        for _, row in transfers.iterrows():
            matrix[datasets.index(row.source), datasets.index(row.target)] = row[f"improvement_{metric}_percent"]
        matrices.append(matrix)
    limit = max(abs(np.nanmin(m)) for m in matrices)
    limit = max(limit, max(np.nanmax(m) for m in matrices))
    cmap = LinearSegmentedColormap.from_list("transfer_diverging", ["#D98972", "#FAFAFA", "#3D7896"])
    cmap.set_bad("#ECEDEF")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    image = None
    for col, (metric, matrix) in enumerate(zip(("MAE", "RMSE", "MAPE"), matrices)):
        ax = axes[col]
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(4), datasets, rotation=32, ha="right")
        ax.set_yticks(range(4), datasets if col == 0 else [])
        ax.set_xlabel("Target network")
        if col == 0:
            ax.set_ylabel("Source network")
        ax.set_title(metric, loc="left", fontweight="bold")
        for i in range(4):
            for j in range(4):
                value = matrix[i, j]
                if np.isfinite(value):
                    color = "white" if abs(value) > 0.52 * limit else INK
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.8, color=color)
                elif i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8.0, color=GREY)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        panel(ax, chr(ord("a") + col), -0.23 if col == 0 else -0.16)
    cax = fig.add_axes([0.93, 0.25, 0.012, 0.54])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Relative error reduction (%)", rotation=270, labelpad=11)
    cbar.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.085, right=0.91, top=0.92, bottom=0.20, wspace=0.30)
    save(fig, "Fig9_cross_network_transfer")


def supplementary_figures() -> None:
    # S1: distribution summaries removed from the main node-level figure.
    nodes = pd.read_csv(DATA / "Figure5_sensor_level_values.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for ax, (dataset, dataset_color) in zip(axes, (("PeMS04", P04), ("PeMS08", P08))):
        d = nodes[nodes.dataset == dataset].copy()
        order = d.network_coherence.rank(method="first")
        d["tertile"] = pd.qcut(order, 3, labels=["Low", "Middle", "High"])
        for xpos, tertile in enumerate(("Low", "Middle", "High")):
            values = d[d.tertile == tertile].node_network_benefit_mae.to_numpy(float)
            med = np.median(values)
            q1, q3 = np.quantile(values, [0.25, 0.75])
            ax.plot([q1, q3], [xpos, xpos], color=dataset_color, lw=3.2, solid_capstyle="round", alpha=0.7)
            ax.scatter([med], [xpos], marker="D", s=22, color=dataset_color, edgecolor="white", lw=0.5)
        ax.axvline(0, color=GREY, lw=0.75, ls="--")
        ax.set_yticks(range(3), ["Low", "Middle", "High"])
        ax.invert_yaxis()
        ax.set_xlabel("Full vs node-only MAE benefit")
        ax.set_title(dataset, loc="left", fontweight="bold")
        clean(ax, xgrid=True)
    axes[0].set_ylabel("Validation coherence tertile")
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.20, wspace=0.30)
    save(fig, "FigS1_node_benefit_median_iqr")

    # S2: sample efficiency plus coefficient trajectories.
    prefix = pd.read_csv(DATA / "Figure7_calibration_prefix.csv")
    coefficients = pd.read_csv(DATA / "Figure7_router_coefficients.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.70), gridspec_kw={"width_ratios": [1.05, 1, 1]})
    for dataset, color, marker in (("PeMS04", P04, "o"), ("PeMS08", P08, "s")):
        d = prefix[prefix.dataset == dataset].sort_values("prefix_percent")
        axes[0].plot(d.prefix_percent, d.MAE_improvement_percent, color=color, marker=marker,
                     ms=3.2, lw=1.1, mfc="white", mew=0.7, label=dataset)
    axes[0].axhline(0, color=GREY, lw=0.75)
    axes[0].set_xticks([5, 10, 20, 50, 100])
    axes[0].set_xlabel("Validation prefix (%)")
    axes[0].set_ylabel("MAE improvement (%)")
    axes[0].set_title("Coefficient fitting", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    clean(axes[0], ygrid=True)
    styles = {
        "node_daily": (P04, "o", "Node daily"),
        "network_daily": (P08, "s", "Network daily"),
        "node_weekly": (MAPE, "^", "Node weekly"),
        "network_weekly": (GOLD, "D", "Network weekly"),
    }
    for ax, dataset in zip(axes[1:], ("PeMS04", "PeMS08")):
        for feature, (color, marker, label) in styles.items():
            d = coefficients[(coefficients.dataset == dataset) & (coefficients.feature == feature)].sort_values("horizon")
            ax.plot(d.horizon, d.coefficient, color=color, marker=marker, ms=2.8, lw=1.0,
                    mfc="white", mew=0.6, label=label)
        ax.axhline(0, color=GREY, lw=0.75)
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.set_xlabel("Forecast horizon")
        ax.set_title(dataset, loc="left", fontweight="bold")
        clean(ax, ygrid=True)
    axes[1].set_ylabel("Routing coefficient")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.66, 1.01))
    for value, ax in zip("abc", axes):
        panel(ax, value, -0.20 if value == "a" else -0.16)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.20, wspace=0.34)
    save(fig, "FigS2_calibration_and_coefficients")

    # S3: evidence-channel and relative-metric sensitivity retained outside the main narrative.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45))
    methods = ["Full CMRR", "Node seasonal only", "Global seasonal only",
               "No magnitude weighting", "Shared coefficients across horizons", "Ordinary ridge"]
    short = ["Full", "Node periodic", "Network periodic", "No magnitude", "Shared beta", "Ridge"]
    for row, dataset in enumerate(("PeMS04", "PeMS08")):
        rel = pd.read_csv(DATA / f"Figure6_{dataset}_relative_metrics.csv")
        base = rel[rel.method == "Frozen backbone"].iloc[0]
        abl = pd.read_csv(DATA / f"Figure6_{dataset}_mechanism_ablation.csv").set_index("method").loc[methods]
        y = np.arange(len(methods))
        for metric, offset in (("MAE", 0.18), ("RMSE", 0.0), ("MAPE", -0.18)):
            gains = 100.0 * (float(base[metric]) - abl[metric].to_numpy(float)) / float(base[metric])
            axes[row, 0].scatter(gains, y + offset, color=METRIC_COLOR[metric], marker=METRIC_MARKER[metric],
                                 s=17, label=metric if row == 0 else None)
        axes[row, 0].set_yticks(y, short)
        axes[row, 0].invert_yaxis()
        axes[row, 0].set_xlabel("Improvement over frozen (%)")
        axes[row, 0].set_title(f"{dataset}: evidence/objective controls", loc="left", fontweight="bold")
        clean(axes[row, 0], xgrid=True)

        measures = ["MAPE", "sMAPE", "node_normalized_MAE_percent", "MAPE_y_gt_10"]
        labels = ["MAPE", "sMAPE", "NN-MAE", "MAPE y>10"]
        x = np.arange(len(measures))
        for method, color, marker in (("Expanding online ridge", GREY, "o"), ("CMRR", P08, "s")):
            values = rel[rel.method == method].iloc[0][measures].to_numpy(float)
            axes[row, 1].plot(x, values, color=color, marker=marker, ms=3.2, lw=1.0,
                              mfc="white", mew=0.7, label=method if row == 0 else None)
        axes[row, 1].set_xticks(x, labels)
        axes[row, 1].set_ylabel("Error (%)")
        axes[row, 1].set_title(f"{dataset}: relative metrics", loc="left", fontweight="bold")
        clean(axes[row, 1], ygrid=True)
    metric_handles, metric_labels = axes[0, 0].get_legend_handles_labels()
    method_handles, method_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(metric_handles, metric_labels, frameon=False, ncols=3, loc="upper center",
               bbox_to_anchor=(0.33, 0.995), columnspacing=1.2, handlelength=1.8)
    fig.legend(method_handles, method_labels, frameon=False, ncols=2, loc="upper center",
               bbox_to_anchor=(0.77, 0.995), columnspacing=1.2, handlelength=1.8)
    for value, ax in zip("abcd", axes.ravel()):
        panel(ax, value, -0.22 if value in "ac" else -0.16)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.87, bottom=0.12, hspace=0.45, wspace=0.34)
    save(fig, "FigS3_evidence_and_metric_sensitivity")

    # S4: cold-start evidence separated from routing-objective evidence.
    cold = pd.read_csv(DATA / "cold_start_recovery.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for ax, dataset in zip(axes, ("PeMS04", "PeMS08")):
        d = cold[(cold.dataset == dataset) & (cold.metric == "MAE")]
        for initialization, color, marker in (("Warm start", P08, "o"), ("Cold start", MAPE, "s")):
            q = d[d.initialization == initialization].sort_values("days")
            ax.plot(q.days, q.improvement_percent, color=color, marker=marker, ms=3.2, lw=1.1,
                    mfc="white", mew=0.7, label=initialization)
        ax.axhline(0, color=GREY, lw=0.75)
        ax.set_xticks(range(1, 8))
        ax.set_xlabel("Deployment day")
        ax.set_ylabel("Cumulative MAE improvement (%)")
        ax.set_title(dataset, loc="left", fontweight="bold")
        clean(ax, ygrid=True)
    axes[0].legend(frameon=False)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.20, wspace=0.30)
    save(fig, "FigS4_cold_start_recovery")


def main() -> None:
    setup()
    figure3()
    figure4()
    figure6()
    figure7()
    figure8()
    figure9()
    supplementary_figures()


if __name__ == "__main__":
    main()
