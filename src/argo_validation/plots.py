"""Phase 26 — ARGO validation plots: RMSE/MAE/bias/correlation and coverage vs depth."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from src.data.loaders.errors import MissingDependencyError

if TYPE_CHECKING:  # pragma: no cover
    from src.argo_validation.pipeline import ArgoValidationResult

PathLike = Union[str, Path]


def plot_argo_validation(result: "ArgoValidationResult", path: PathLike) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError("matplotlib", "ARGO validation plots") from exc

    report = result.report
    per_depth = (report.get("metrics") or report.get("pipeline_check_metrics") or {}).get("per_depth", {})
    coverage = report["depth_coverage"]["per_depth"]
    depths = np.asarray(result.depths_m, dtype=float)
    labels = list(per_depth.keys())

    def series(key: str, source=per_depth):
        return np.array([np.nan if source[l][key] is None else source[l][key] for l in labels], dtype=float)

    panels = [("RMSE (degC)", series("rmse")), ("MAE (degC)", series("mae")),
              ("Bias NEER-ARGO (degC)", series("bias")), ("Correlation (per depth)", series("correlation")),
              ("Profiles with value", np.array([coverage[l]["n_profiles"] for l in labels], dtype=float))]

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5), sharey=True)
    for ax, (title, values) in zip(axes, panels):
        ax.plot(values, depths, marker="o")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if "Bias" in title:
            ax.axvline(0.0, color="k", lw=0.8)
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()

    if result.is_demo:
        head = "[DEMO — SYNTHETIC ARGO, NOT OBSERVATIONS — pipeline check only]"
        fig.patch.set_facecolor("#fff4f4")
        fig.text(0.5, 0.005, head, ha="center", color="crimson", fontsize=11, weight="bold")
        fig.suptitle(head, color="crimson", fontsize=12)
    else:
        fig.suptitle("NEER vs ARGO profiles", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path