"""
Preprocessing metadata: the record of how a tensor file came to exist.

A `.npz` of normalized tensors is not self-explanatory. Six months from
now, "why is channel 3 negative?" and "were these the statistics the
checkpoint was trained with?" are questions someone will have to answer,
and the arrays alone cannot answer them. `PreprocessingMetadata` is
written alongside every tensor file and records:

* **provenance** — source file, `data_mode`, and whether the data is
  synthetic, carried through from the loader so demo tensors can never
  be mistaken for real ones;
* **the pipeline** — every step in order, with its configuration, its
  fitted state, and what it reported doing (how many cells each fill
  strategy touched, coverage per variable, ...);
* **the split and the fit window** — the chronological boundaries, the
  per-period counts, and the exact training time range the fitted steps
  saw, so the no-leakage claim can be checked after the fact rather than
  taken on faith;
* **the tensors** — shapes, channel order, dtype, fill value, valid
  fractions.

The normalization statistics are stored in full, deliberately: they are
small, and they are what a later phase needs to convert predictions back
into degrees Celsius. Everything else is summarized — a climatology field
belongs in the tensor file, not in a JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.data.preprocessing._utils import json_safe

PathLike = Union[str, Path]

#: Bumped when the meaning of the pipeline's output changes, so a stale
#: tensor file can be recognized rather than silently reused.
PREPROCESSING_VERSION = "1.0.0"


@dataclass
class PreprocessingMetadata:
    """Everything needed to reproduce and interpret a preprocessing run."""

    pipeline: str
    version: str = PREPROCESSING_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    project: Dict[str, Any] = field(default_factory=dict)
    environment: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    split: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, Any] = field(default_factory=dict)
    leakage: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # -- access ------------------------------------------------------------

    def step(self, name: str) -> Optional[Dict[str, Any]]:
        """The recorded entry for one step, by its `name`."""
        for entry in self.steps:
            if entry.get("step") == name:
                return entry
        return None

    @property
    def is_synthetic(self) -> bool:
        return bool(self.source.get("is_synthetic"))

    def normalization_statistics(self) -> Dict[str, Any]:
        """The fitted normalization statistics, for inverting predictions."""
        entry = self.step("normalization") or {}
        return entry.get("state", {}).get("statistics", {})

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(
            {
                "pipeline": self.pipeline,
                "version": self.version,
                "created_at": self.created_at,
                "project": self.project,
                "environment": self.environment,
                "source": self.source,
                "split": self.split,
                "leakage": self.leakage,
                "steps": self.steps,
                "tensors": self.tensors,
                "outputs": self.outputs,
                "notes": self.notes,
            }
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        """Human-readable summary, for the console and for logs."""
        lines = [
            "NEER preprocessing metadata",
            "=" * 60,
            f"pipeline   : {self.pipeline} (v{self.version})",
            f"created    : {self.created_at}",
            f"source     : {self.source.get('source') or '-'}",
            f"data mode  : {self.source.get('data_mode') or '-'}"
            + ("  ⚠ SYNTHETIC" if self.is_synthetic else ""),
            "",
            "Split",
            "-" * 60,
            f"  strategy : {self.split.get('strategy', '-')}",
            f"  counts   : {self.split.get('counts', {})}",
            f"  ranges   : {self.split.get('ranges', {})}",
            f"  fitted on: {self.split.get('fitted_on', {})}",
        ]
        coverage = self.split.get("spatial_coverage", {})
        if coverage:
            lines.append(
                f"  spatial  : grid={coverage.get('grid_shape')} "
                f"ocean_fraction={coverage.get('ocean_fraction')}"
            )
            for period, stats in coverage.get("by_period", {}).items():
                lines.append(
                    f"             {period:<6} valid_fraction={stats.get('valid_fraction')} "
                    f"cells_ever_observed={stats.get('cells_ever_observed_fraction')}"
                )
        lines += [
            "",
            "Steps",
            "-" * 60,
        ]
        for entry in self.steps:
            marker = "learned" if entry.get("learns_from_data") else "stateless"
            lines.append(f"  {entry.get('step'):<24} [{marker}]")
        if self.leakage:
            lines += ["", "Leakage control", "-" * 60]
            for entry in self.leakage.get("learned_steps", []):
                lines.append(
                    f"  {entry.get('step'):<24} fitted on {entry.get('n_timesteps')} "
                    f"timestep(s), {entry.get('first')} .. {entry.get('last')}"
                )
            periods = self.leakage.get("periods_transformed_independently")
            if periods:
                lines.append(
                    f"  periods transformed independently: {', '.join(periods)}"
                )
        tensors = self.tensors.get("inputs", {})
        if tensors:
            lines += [
                "",
                "Tensors",
                "-" * 60,
                f"  inputs   : {tensors.get('shape')}  {tensors.get('layout')}",
                f"  channels : {', '.join(tensors.get('channels', []))}",
            ]
            targets = self.tensors.get("targets")
            if targets:
                lines.append(f"  targets  : {targets.get('shape')}  {targets.get('layout')}")
        if self.outputs:
            lines += ["", "Outputs", "-" * 60]
            lines += [f"  {key}: {value}" for key, value in self.outputs.items()]
        if self.notes:
            lines += ["", "Notes", "-" * 60]
            lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)

    def save(self, path: PathLike) -> Path:
        """Write the metadata to disk; `.json` gives JSON, anything else text."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.to_json() if path.suffix.lower() == ".json" else self.to_text()
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: PathLike) -> "PreprocessingMetadata":
        """Read metadata written by `save` (JSON only)."""
        with open(Path(path), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingMetadata":
        known = {
            "pipeline",
            "version",
            "created_at",
            "project",
            "environment",
            "source",
            "steps",
            "split",
            "tensors",
            "leakage",
            "outputs",
            "notes",
        }
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.to_text()
