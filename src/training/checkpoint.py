"""
NEER — Phase 27 checkpoint manager.

A checkpoint is more than a `state_dict`: reproducing or resuming a run
needs the optimizer state, which epoch it's from, the config and seed it
was trained with, and what code produced it. This module is the one place
that builds and reads that bundle, so every checkpoint in the project —
`scripts/train.py`'s and any future training script's — has the same
shape and the same compatibility checking.

What's stored, every time
--------------------------
* ``model_state_dict`` / ``optimizer_state_dict`` — the tensors.
* ``metadata`` (JSON-safe, also mirrored as a ``.json`` sidecar so it can
  be inspected without touching torch or the weights):

  - ``epoch``, ``seed``
  - ``config`` — the resolved config's raw dict (``NeerConfig.raw``, a
    plain dict, or a dataclass), plus a short ``config_fingerprint`` hash
    of the architecture-relevant fields (``model.embedding_dim``,
    ``model.use_gnn``, ``uncertainty.enabled``, ``depths``,
    ``environment``) for a fast human-readable mismatch check
  - ``training_stats`` — whatever the caller passes (loss curves, best
    val loss, ...), sanitized to plain JSON types
  - ``model`` — class name, module, parameter count
  - ``state_dict_shapes`` — ``{param_name: shape}`` for every tensor in
    the checkpoint; this, not the fingerprint, is what `check_compatibility`
    actually loads against
  - ``git`` — commit/branch/dirty/remote, gathered with the `git` CLI;
    ``{"available": False, "reason": ...}`` outside a repo or without git
    installed, never an exception
  - ``environment_info`` — python/torch/numpy version, platform
  - ``extra`` — anything else the caller wants attached

Validating compatibility before loading
-----------------------------------------
`load_checkpoint` (and `CheckpointManager.load_checkpoint`) always calls
`check_compatibility` first. Against a `model`, it diffs
`state_dict_shapes`: a shape mismatch is always an error (torch would
refuse it too, with a less specific message); missing/unexpected keys are
errors under `strict=True` (the default, matching `nn.Module.load_state_dict`'s
own default) and warnings under `strict=False`. Against a `config`, a
`config_fingerprint` mismatch is a warning, not an error — the state-dict
shape check is authoritative for whether it will actually load. On a
schema/shape error, `CheckpointCompatibilityError` is raised *before*
`load_state_dict` runs, so a bad checkpoint fails with a specific report,
not a generic torch RuntimeError. Warnings are logged, not raised.
Checkpoints from before this module existed load too, as a `"legacy"`
schema with an empty `state_dict_shapes`/`git`/`config_fingerprint` (so
they carry no shape-mismatch guarantees) — the loader still works, it
just cannot validate them as closely.

Usage
-----
    from src.training.checkpoint import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir, name="neer")
    mgr.save_last(model=model, optimizer=optimizer, epoch=epoch, seed=seed,
                  config=config, training_stats={"val_loss": val_loss, "history": history})
    if improved:
        mgr.save_best(model=model, optimizer=optimizer, epoch=epoch, seed=seed,
                      config=config, training_stats={"val_loss": val_loss})
    ...
    loaded = mgr.load_checkpoint(mgr.last_path, model=model, optimizer=optimizer, config=config)
    print(loaded.metadata["epoch"], loaded.report.warnings)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass, field, is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.data.loaders.errors import MissingDependencyError
from src.utils.logging import get_logger

try:  # pragma: no cover - environment dependent
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = get_logger("neer.checkpoint")

PathLike = Union[str, Path]

#: Bumped when the metadata *shape* changes in a way old readers can't parse.
#: Checkpoints without a `metadata` key at all (pre-Phase-27) load as `"legacy"`.
SCHEMA_VERSION = 1
LEGACY_SCHEMA = "legacy"

DEFAULT_CHECKPOINT_DIR = Path("artifacts") / "checkpoints"
DEFAULT_NAME = "neer"

#: Config keys that determine whether a checkpoint's tensors can even
#: shape-match a freshly built model; used only for the fast, informational
#: `config_fingerprint` — `state_dict_shapes` is the real compatibility check.
_FINGERPRINT_FIELDS = ("environment", "model.embedding_dim", "model.use_gnn", "uncertainty.enabled", "depths")


class CheckpointError(Exception):
    """Base class for checkpoint save/load failures."""


class CheckpointCompatibilityError(CheckpointError):
    """A checkpoint failed `check_compatibility` before any weights were touched."""

    def __init__(self, report: "CompatibilityReport"):
        self.report = report
        super().__init__("checkpoint is not compatible:\n  " + "\n  ".join(report.errors))


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise MissingDependencyError(
            package="torch", purpose="saving or loading NEER checkpoints", install_hint="pip install torch"
        )


# --------------------------------------------------------------------------
# JSON-safety and config handling
# --------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Recursively coerce `obj` into plain JSON-serializable types.

    Handles numpy scalars/arrays, `Path`, dataclasses, and (when torch is
    available) 0-d/1-element torch tensors; anything else unrecognized
    falls back to `str(obj)` rather than raising, so an odd value in
    `training_stats` or `extra` never breaks a save.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if _TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        return _json_safe(obj.detach().cpu().tolist())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Accept a `NeerConfig` (has `.raw`), a plain dict, a dataclass, or `None`."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return _json_safe(config)
    if hasattr(config, "raw") and isinstance(getattr(config, "raw"), dict):
        d = dict(config.raw)
        d.setdefault("environment", getattr(config, "environment", d.get("environment")))
        return _json_safe(d)
    if is_dataclass(config):
        return _json_safe(asdict(config))
    raise TypeError(f"config must be a NeerConfig, dict, or dataclass; got {type(config)}")


def _dig(d: Dict[str, Any], dotted_key: str) -> Any:
    node: Any = d
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def config_fingerprint(config_dict: Dict[str, Any]) -> str:
    """Short hash of the architecture-relevant fields of a config dict.

    Two configs that would build differently-shaped models almost always
    differ here — but this is a *fast, human-readable* early warning, not
    the compatibility check itself; `state_dict_shapes` is authoritative."""
    fields = {k: _dig(config_dict, k) for k in _FINGERPRINT_FIELDS}
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest[:16]


# --------------------------------------------------------------------------
# Git / project metadata
# --------------------------------------------------------------------------


def _run_git(args: Sequence[str], cwd: Optional[Path]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
    )
    return result.stdout.strip()


def git_metadata(cwd: Optional[PathLike] = None) -> Dict[str, Any]:
    """Best-effort git commit/branch/dirty/remote for the repo containing `cwd`
    (default: this file's location). Never raises: outside a git repo, with
    git not installed, or on any other failure, returns
    ``{"available": False, "reason": ...}`` so a checkpoint can always be saved."""
    cwd = Path(cwd) if cwd is not None else Path(__file__).resolve().parent
    try:
        commit = _run_git(["rev-parse", "HEAD"], cwd)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        status = _run_git(["status", "--porcelain"], cwd)
        try:
            remote = _run_git(["remote", "get-url", "origin"], cwd)
        except (subprocess.CalledProcessError, OSError):
            remote = None
        return {
            "available": True,
            "commit": commit,
            "commit_short": commit[:12],
            "branch": branch,
            "dirty": bool(status),
            "remote": remote,
        }
    except FileNotFoundError:
        return {"available": False, "reason": "git executable not found"}
    except subprocess.CalledProcessError as exc:
        return {"available": False, "reason": f"not a git repository or git error: {exc.stderr.strip() or exc}"}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "git command timed out"}
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def environment_info() -> Dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None) if _TORCH_AVAILABLE else None,
        "numpy_version": np.__version__,
    }


# --------------------------------------------------------------------------
# Metadata / compatibility report
# --------------------------------------------------------------------------


@dataclass
class CompatibilityReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)   # in model, not in checkpoint
    unexpected_keys: List[str] = field(default_factory=list)  # in checkpoint, not in model
    shape_mismatches: List[str] = field(default_factory=list)

    def raise_if_incompatible(self) -> None:
        if not self.ok:
            raise CheckpointCompatibilityError(self)


def _state_dict_shapes(model: Any) -> Dict[str, List[int]]:
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def check_compatibility(
    metadata: Dict[str, Any],
    *,
    model: Optional[Any] = None,
    config: Optional[Any] = None,
    strict: bool = True,
) -> CompatibilityReport:
    """Compare a checkpoint's `metadata` against a live `model` and/or `config`
    *before* any weights are loaded. See the module docstring for what counts
    as an error vs. a warning."""
    errors: List[str] = []
    warnings: List[str] = []
    missing: List[str] = []
    unexpected: List[str] = []
    mismatched: List[str] = []

    schema = metadata.get("schema_version")
    if schema not in (SCHEMA_VERSION, LEGACY_SCHEMA):
        errors.append(
            f"unsupported checkpoint schema_version {schema!r} "
            f"(this code reads {SCHEMA_VERSION} and {LEGACY_SCHEMA!r})"
        )
    if schema == LEGACY_SCHEMA:
        warnings.append(
            "legacy checkpoint (no Phase-27 metadata): no state_dict_shapes or git info to validate against; "
            "loading will still be attempted."
        )

    if model is not None:
        current_shapes = _state_dict_shapes(model)
        stored_shapes = metadata.get("state_dict_shapes") or {}
        if stored_shapes:
            missing = sorted(set(current_shapes) - set(stored_shapes))
            unexpected = sorted(set(stored_shapes) - set(current_shapes))
            mismatched = [
                f"{k}: model expects {current_shapes[k]}, checkpoint has {stored_shapes[k]}"
                for k in sorted(set(current_shapes) & set(stored_shapes))
                if list(current_shapes[k]) != list(stored_shapes[k])
            ]
            if mismatched:
                errors.append(f"{len(mismatched)} parameter shape mismatch(es): " + "; ".join(mismatched))
            if missing:
                msg = f"{len(missing)} parameter(s) the model has are missing from the checkpoint: {missing}"
                (errors if strict else warnings).append(msg)
            if unexpected:
                msg = f"{len(unexpected)} parameter(s) in the checkpoint are not in the model: {unexpected}"
                (errors if strict else warnings).append(msg)

    if config is not None:
        current_fp = config_fingerprint(_config_to_dict(config))
        stored_fp = metadata.get("config_fingerprint")
        if stored_fp and stored_fp != current_fp:
            warnings.append(
                f"config fingerprint differs from the checkpoint's ({stored_fp} vs {current_fp}); "
                "it may have been trained with different architecture settings "
                "(embedding_dim / use_gnn / uncertainty.enabled / depths / environment) — "
                "a real mismatch will surface as a shape error if a model is also checked."
            )

    return CompatibilityReport(
        ok=not errors, errors=errors, warnings=warnings,
        missing_keys=missing, unexpected_keys=unexpected, shape_mismatches=mismatched,
    )


def build_metadata(
    *,
    kind: str,
    epoch: int,
    model: Optional[Any] = None,
    seed: Optional[int] = None,
    config: Optional[Any] = None,
    training_stats: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config_dict = _config_to_dict(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "seed": None if seed is None else int(seed),
        "config": config_dict,
        "config_fingerprint": config_fingerprint(config_dict) if config_dict else None,
        "training_stats": _json_safe(training_stats or {}),
        "model": {
            "class_name": type(model).__name__ if model is not None else None,
            "module": type(model).__module__ if model is not None else None,
            "num_parameters": (
                int(sum(p.numel() for p in model.parameters())) if model is not None else None
            ),
        },
        "state_dict_shapes": _state_dict_shapes(model) if model is not None else {},
        "git": git_metadata(),
        "environment_info": environment_info(),
        "extra": _json_safe(extra or {}),
    }


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------


@dataclass
class LoadedCheckpoint:
    path: Path
    metadata: Dict[str, Any]
    report: CompatibilityReport
    raw: Dict[str, Any]  # the full torch.load()'d dict, incl. model_state_dict / optimizer_state_dict

    @property
    def epoch(self) -> int:
        return int(self.metadata.get("epoch", 0))

    @property
    def training_stats(self) -> Dict[str, Any]:
        return self.metadata.get("training_stats", {})


def _normalize_legacy(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize a minimal metadata block for a pre-Phase-27 checkpoint
    (one with `epoch`/`model_state_dict`/... at the top level and no
    `metadata` key), so it still loads through this module."""
    return {
        "schema_version": LEGACY_SCHEMA,
        "kind": "unknown",
        "saved_at": None,
        "epoch": raw.get("epoch", 0),
        "seed": None,
        "config": {},
        "config_fingerprint": None,
        "training_stats": {
            k: raw[k] for k in ("val_loss", "best_val_loss", "history") if k in raw
        },
        "model": {"class_name": None, "module": None, "num_parameters": None},
        "state_dict_shapes": {},
        "git": {"available": False, "reason": "checkpoint predates Phase 27 metadata"},
        "environment_info": {},
        "extra": {"legacy_args": raw.get("args")},
    }


def save_checkpoint(
    path: PathLike,
    *,
    model: Any,
    epoch: int,
    optimizer: Optional[Any] = None,
    seed: Optional[int] = None,
    config: Optional[Any] = None,
    training_stats: Optional[Dict[str, Any]] = None,
    kind: str = "manual",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one checkpoint. Atomic: builds the file next to `path` under a
    temp name and renames it into place, so a crash or a full disk mid-write
    never leaves a truncated file at `path`. Also writes a `<path>.json`
    sidecar with just the (torch-free, human-readable) metadata."""
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(
        kind=kind, epoch=epoch, model=model, seed=seed, config=config,
        training_stats=training_stats, extra=extra,
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata,
    }

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)

    sidecar = path.with_suffix(path.suffix + ".json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path


def load_checkpoint(
    path: PathLike,
    *,
    model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    config: Optional[Any] = None,
    map_location: Union[str, "torch.device"] = "cpu",
    strict: bool = True,
    validate: bool = True,
) -> LoadedCheckpoint:
    """Load one checkpoint, validating it first (see `check_compatibility`).

    If `model`/`optimizer` are given, their state is loaded into them
    in-place (after validation passes). `validate=False` skips the
    "raise on error" step (useful for e.g. an inspection CLI) but the
    `CompatibilityReport` is still returned so the caller can inspect it.

    Trust note: like the rest of this codebase's checkpoint handling,
    this uses `torch.load(..., weights_only=False)` — appropriate for
    checkpoints this project produced, not for arbitrary `.pt` files
    from an untrusted source.
    """
    _require_torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    raw = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        raise CheckpointError(f"{path} does not look like a NEER checkpoint (no 'model_state_dict')")

    metadata = raw.get("metadata")
    if metadata is None:
        logger.warning("%s has no Phase-27 metadata; treating it as a legacy checkpoint.", path)
        metadata = _normalize_legacy(raw)

    report = check_compatibility(metadata, model=model, config=config, strict=strict)
    for w in report.warnings:
        logger.warning("checkpoint %s: %s", path, w)
    if validate:
        report.raise_if_incompatible()

    if model is not None:
        model.load_state_dict(raw["model_state_dict"], strict=strict)
    if optimizer is not None and raw.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(raw["optimizer_state_dict"])
    elif optimizer is not None:
        logger.warning("checkpoint %s has no optimizer state; optimizer left unchanged.", path)

    return LoadedCheckpoint(path=path, metadata=metadata, report=report, raw=raw)


# --------------------------------------------------------------------------
# Stateful convenience wrapper
# --------------------------------------------------------------------------


class CheckpointManager:
    """Owns one `checkpoint_dir` and the `{name}_best.pt` / `{name}_last.pt`
    pair in it — the thing a training loop calls every epoch.

    >>> mgr = CheckpointManager("artifacts/checkpoints", name="neer")
    >>> mgr.save_last(model=model, optimizer=optimizer, epoch=epoch, seed=seed, config=config,
    ...                training_stats={"val_loss": val_loss})   # doctest: +SKIP
    >>> if improved:
    ...     mgr.save_best(model=model, optimizer=optimizer, epoch=epoch, seed=seed,
    ...                   config=config, training_stats={"val_loss": val_loss})  # doctest: +SKIP
    """

    def __init__(self, checkpoint_dir: PathLike = DEFAULT_CHECKPOINT_DIR, name: str = DEFAULT_NAME):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.name = name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / f"{self.name}_best.pt"

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / f"{self.name}_last.pt"

    def path_for(self, tag: str) -> Path:
        return self.checkpoint_dir / f"{self.name}_{tag}.pt"

    def save(self, path: PathLike, *, kind: str = "manual", **kwargs: Any) -> Path:
        return save_checkpoint(path, kind=kind, **kwargs)

    def save_best(self, **kwargs: Any) -> Path:
        return self.save(self.best_path, kind="best", **kwargs)

    def save_last(self, **kwargs: Any) -> Path:
        return self.save(self.last_path, kind="last", **kwargs)

    def load_checkpoint(self, path: Optional[PathLike] = None, **kwargs: Any) -> LoadedCheckpoint:
        return load_checkpoint(self.last_path if path is None else path, **kwargs)

    def load_best(self, **kwargs: Any) -> LoadedCheckpoint:
        return load_checkpoint(self.best_path, **kwargs)

    def load_last(self, **kwargs: Any) -> LoadedCheckpoint:
        return load_checkpoint(self.last_path, **kwargs)

    def list_checkpoints(self) -> List[Path]:
        return sorted(self.checkpoint_dir.glob(f"{self.name}_*.pt"))

    def __repr__(self) -> str:
        return f"CheckpointManager(checkpoint_dir={self.checkpoint_dir!s}, name={self.name!r})"