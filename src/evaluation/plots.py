"""
Phase 25 — evaluation plots.

Turns the `per_depth` breakdown inside a `src.evaluation.metrics.ProfileMetrics`
(equivalently, `src.evaluation.interface.EvaluationReport.metrics[split]`)
into the four plots the evaluation engine is asked for: RMSE, MAE, bias,
and Pearson correlation, each against depth.

One call, `plot_metric_vs_depth`, draws one of the four metrics for one
or more models on one split, so a single figure can either show one
model's error profile or overlay several models (e.g. every baseline
plus NEER) for a direct depth-by-depth comparison. `plot_all_metrics`
is the convenience wrapper the evaluation engine actually calls: it
produces all four plots for a split in one pass and saves them to disk.

Depth is plotted on the y-axis, increasing downward (oceanographic
convention — surface at the top), with the metric on the x-axis, since
that is the more familiar way to read a depth profile than the reverse.

Synthetic data
--------------
Every plot this module draws accepts an `is_synthetic` flag. When set,
the figure's title is prefixed with a `[SYNTHETIC DEMO DATA]` banner
and a matching footnote is added in the corner of the axes, so a saved
PNG can never be mistaken for a plot of real observations even once it
has been separated from the JSON report or file path it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

from src.evaluation.metrics import MetricSet, ProfileMetrics

PathLike = Union[str, Path]

#: The four metrics Phase 25 plots against depth, and how each is
#: labeled on an axis.
METRIC_LABELS: Dict[str, str] = {
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "Bias (pred - target)",
    "pearson": "Pearson correlation (r)",
}

SYNTHETIC_BANNER = "[SYNTHETIC DEMO DATA]"
SYNTHETIC_FOOTNOTE = (
    "Computed on the NEER synthetic demo dataset (data_mode=DEMO_SYNTHETIC) — "
    "not real observations. For pipeline demonstration only."
)


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe: no display required to save PNGs
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        from src.data.loaders.errors import MissingDependencyError

        raise MissingDependencyError(
            package="matplotlib",
            purpose="Phase 25 evaluation plots (src.evaluation.plots)",
            install_hint="pip install matplotlib",
        ) from exc
    return plt


def _depth_values(depth_names: Sequence[str]) -> np.ndarray:
    """Best-effort numeric depth axis from labels like `"0m"`, `"50m"`.

    Falls back to the column index when a label doesn't parse (e.g. a
    hand-built bundle that labeled depths some other way), so plotting
    never fails outright over a cosmetic axis choice — it just orders
    depths by position instead of by parsed value.
    """
    values = []
    for i, name in enumerate(depth_names):
        try:
            values.append(float(str(name).rstrip("mM")))
        except ValueError:
            values.append(float(i))
    return np.asarray(values, dtype=float)


def _extract_series(
    profile_metrics: ProfileMetrics, metric: str, depth_names: Sequence[str]
) -> np.ndarray:
    """`metric` values per depth, in `depth_names` order; `nan` where a
    depth's `MetricSet` reported `None` (no valid data), so a broken
    line in the plot honestly shows a gap rather than a fabricated 0."""
    values = np.full(len(depth_names), np.nan, dtype=float)
    for i, name in enumerate(depth_names):
        metric_set: Optional[MetricSet] = profile_metrics.per_depth.get(name)
        if metric_set is None:
            continue
        value = getattr(metric_set, metric)
        if value is not None:
            values[i] = value
    return values


def plot_metric_vs_depth(
    metric: str,
    depth_names: Sequence[str],
    series: Mapping[str, ProfileMetrics],
    *,
    split: str = "test",
    is_synthetic: bool = False,
    ax=None,
):
    """Draw `metric` (one of `"rmse"`, `"mae"`, `"bias"`, `"pearson"`)
    against depth for one or more models.

    `series` maps a model name to its `ProfileMetrics` for `split` — a
    single-entry dict plots one model, a multi-entry dict overlays them
    (e.g. `{"climatology": ..., "ridge": ..., "neer": ...}`) so models
    can be compared depth-by-depth on the same axes.

    Returns the `matplotlib.figure.Figure` (new, unless `ax` was given,
    in which case the figure `ax` belongs to).
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"unknown metric '{metric}'; expected one of {list(METRIC_LABELS)}")

    plt = _require_matplotlib()
    depth_values = _depth_values(depth_names)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    for model_name, profile_metrics in series.items():
        values = _extract_series(profile_metrics, metric, depth_names)
        ax.plot(values, depth_values, marker="o", label=model_name)

    if metric == "bias":
        ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")

    ax.invert_yaxis()  # surface (shallow) at the top, like a real depth profile
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_ylabel("Depth (m)")
    title = f"{METRIC_LABELS[metric]} vs depth — {split}"
    if is_synthetic:
        title = f"{SYNTHETIC_BANNER} {title}"
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if owns_figure:
        fig.tight_layout()

    if is_synthetic and owns_figure:
        fig.subplots_adjust(bottom=0.16)
        fig.text(
            0.5, 0.02, SYNTHETIC_FOOTNOTE, fontsize=7, color="firebrick", ha="center", wrap=True
        )

    return fig


def plot_all_metrics(
    depth_names: Sequence[str],
    series: Mapping[str, ProfileMetrics],
    output_dir: PathLike,
    *,
    split: str = "test",
    is_synthetic: bool = False,
    prefix: Optional[str] = None,
) -> Dict[str, Path]:
    """Save all four depth plots (RMSE, MAE, bias, Pearson correlation)
    for `split` to `output_dir`, one PNG each.

    `series` is the same `{model_name: ProfileMetrics}` mapping
    `plot_metric_vs_depth` takes — pass every model being compared so
    each PNG overlays all of them.

    Returns `{metric_name: saved_path}`.
    """
    plt = _require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{prefix}_" if prefix else ""

    saved: Dict[str, Path] = {}
    for metric in METRIC_LABELS:
        fig = plot_metric_vs_depth(
            metric, depth_names, series, split=split, is_synthetic=is_synthetic
        )
        path = output_dir / f"{file_prefix}{metric}_vs_depth_{split}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved[metric] = path
    return saved