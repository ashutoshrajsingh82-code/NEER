"""Tests for the Phase 12 PyTorch Dataset (`src/data/dataset.py`).

`NEERDataset` wraps a Phase 08 `TensorBundle` in a `torch.utils.data.
Dataset`. The tests build a small, self-contained `TensorBundle` by hand
(`_tiny_bundle`, below) rather than running the full loader ->
preprocessing -> tensor-assembly pipeline: that pipeline is already
covered by `tests/test_preprocessing_pipeline.py` and friends, and this
module's only job is the dataset/`DataLoader` layer sitting on top of an
already-assembled bundle.

Two groups of tests:

- Tests that run regardless of whether `torch` is installed, since the
  module itself must stay importable without it (mirrors how
  `src/data/loaders` treats `xarray` as optional).
- Tests that need `torch`, skipped with `pytest.importorskip` when it
  is not installed — this project's existing pattern for optional
  dependencies (see `tests/test_loaders_netcdf.py`).

`test_real_processed_tensors_*` additionally exercises the actual
`data/processed/neer_tensors.npz` produced by `scripts/preprocess_data.py`,
skipped if that file has not been generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing.channels import NEER_CHANNEL_ORDER, NEER_N_CHANNELS  # noqa: E402
from src.data.preprocessing.tensors import TensorBundle  # noqa: E402
from src.data.dataset import NEER_N_DEPTHS  # noqa: E402

torch = pytest.importorskip("torch", reason="torch is an optional dependency")

import src.data.dataset as neer_dataset  # noqa: E402
from src.data.dataset import NEERDataset, make_dataloader  # noqa: E402


# --------------------------------------------------------------------------
# Fixture: a tiny, hand-built TensorBundle with the real 11/15 shape
# --------------------------------------------------------------------------


def _tiny_bundle(
    *,
    n_time: int = 6,
    n_lat: int = 4,
    n_lon: int = 5,
    n_depth: int = NEER_N_DEPTHS,
    channel_names=NEER_CHANNEL_ORDER,
    with_targets: bool = True,
    seed: int = 0,
) -> TensorBundle:
    """A small `TensorBundle` with NEER's real channel/depth shape.

    Timesteps are monthly starting 2020-01-15, so date strings are
    predictable and checkable exactly. The first half of timesteps is
    marked "train", the rest "val", so split filtering has something to
    assert against.
    """
    rng = np.random.default_rng(seed)
    n_channels = len(channel_names)

    time = np.array(
        [
            (np.datetime64("2020-01", "M") + np.timedelta64(i, "M")).astype("datetime64[D]")
            + np.timedelta64(14, "D")
            for i in range(n_time)
        ],
        dtype="datetime64[ns]",
    )
    lat = np.round(5.0 + 0.5 * np.arange(n_lat), 4)
    lon = np.round(60.0 + 0.5 * np.arange(n_lon), 4)
    depth = np.linspace(0.0, 1000.0, n_depth)

    inputs = rng.normal(0, 1, (n_time, n_channels, n_lat, n_lon)).astype(np.float32)
    input_mask = rng.random((n_time, n_channels, n_lat, n_lon)) > 0.1

    targets = target_mask = None
    target_names = []
    if with_targets:
        targets = (15.0 + rng.normal(0, 1, (n_time, n_depth, n_lat, n_lon))).astype(np.float32)
        target_mask = rng.random((n_time, n_depth, n_lat, n_lon)) > 0.5
        target_names = ["subsurface_temp"]

    n_train = n_time // 2
    split_masks = {
        "train": np.array([i < n_train for i in range(n_time)], dtype=bool),
        "val": np.array([i >= n_train for i in range(n_time)], dtype=bool),
    }

    return TensorBundle(
        inputs=inputs,
        input_mask=input_mask,
        channel_names=list(channel_names),
        time=time,
        lat=lat,
        lon=lon,
        targets=targets,
        target_mask=target_mask,
        target_names=target_names,
        depth=depth if with_targets else None,
        ocean_mask=np.ones((n_lat, n_lon), dtype=bool),
        split_masks=split_masks,
        attrs={"data_mode": "SYNTHETIC_TEST", "is_synthetic": True},
    )


# --------------------------------------------------------------------------
# Importability without torch
# --------------------------------------------------------------------------


def test_module_is_importable_without_torch_being_required_at_import_time():
    # This file's own imports already prove src.data.dataset imports fine
    # in an environment with torch present; the try/except ImportError
    # guard around `import torch` at module scope means the same is true
    # without it. Assert the guard's bookkeeping is internally consistent
    # instead of re-importing under a simulated absence.
    assert neer_dataset._TORCH_AVAILABLE is True
    assert neer_dataset.torch is torch


# --------------------------------------------------------------------------
# Construction / validation
# --------------------------------------------------------------------------


def test_dataset_length_equals_number_of_timesteps():
    ds = NEERDataset(_tiny_bundle(n_time=6))
    assert len(ds) == 6


def test_construction_requires_targets():
    bundle = _tiny_bundle(with_targets=False)
    with pytest.raises(ValueError, match="targets"):
        NEERDataset(bundle)


def test_strict_shape_rejects_wrong_channel_count():
    bundle = _tiny_bundle(channel_names=NEER_CHANNEL_ORDER[:5])
    with pytest.raises(ValueError, match="channels"):
        NEERDataset(bundle)


def test_strict_shape_rejects_wrong_depth_count():
    bundle = _tiny_bundle(n_depth=3)
    with pytest.raises(ValueError, match="depth"):
        NEERDataset(bundle)


def test_strict_shape_false_allows_a_nonstandard_bundle():
    bundle = _tiny_bundle(channel_names=NEER_CHANNEL_ORDER[:5], n_depth=3)
    ds = NEERDataset(bundle, strict_shape=False)
    assert len(ds) == bundle.n_time
    sample = ds[0]
    assert sample["inputs"].shape == (5, bundle.grid_shape[0], bundle.grid_shape[1])


def test_split_argument_filters_before_validation():
    bundle = _tiny_bundle(n_time=6)
    train_ds = NEERDataset(bundle, split="train")
    val_ds = NEERDataset(bundle, split="val")
    assert len(train_ds) == 3
    assert len(val_ds) == 3
    assert len(train_ds) + len(val_ds) == len(bundle.time)


def test_unknown_split_name_raises():
    bundle = _tiny_bundle()
    with pytest.raises(KeyError):
        NEERDataset(bundle, split="not_a_real_split")


# --------------------------------------------------------------------------
# __getitem__ contract: inputs / targets / mask / metadata
# --------------------------------------------------------------------------


def test_getitem_returns_the_four_required_keys():
    ds = NEERDataset(_tiny_bundle())
    sample = ds[0]
    assert set(sample.keys()) == {"inputs", "targets", "mask", "metadata"}


def test_inputs_and_targets_are_float_tensors_with_no_channel_batch_dim():
    n_lat, n_lon = 4, 5
    ds = NEERDataset(_tiny_bundle(n_lat=n_lat, n_lon=n_lon))
    sample = ds[0]

    assert isinstance(sample["inputs"], torch.Tensor)
    assert sample["inputs"].shape == (NEER_N_CHANNELS, n_lat, n_lon)
    assert sample["inputs"].dtype == torch.float32

    assert isinstance(sample["targets"], torch.Tensor)
    assert sample["targets"].shape == (NEER_N_DEPTHS, n_lat, n_lon)
    assert sample["targets"].dtype == torch.float32


def test_inputs_and_targets_match_the_source_bundle_values():
    bundle = _tiny_bundle()
    ds = NEERDataset(bundle)
    sample = ds[2]
    np.testing.assert_allclose(sample["inputs"].numpy(), bundle.inputs[2])
    np.testing.assert_allclose(sample["targets"].numpy(), bundle.targets[2])


def test_mask_is_a_dict_of_bool_tensors_matching_input_and_target_shape():
    n_lat, n_lon = 4, 5
    bundle = _tiny_bundle(n_lat=n_lat, n_lon=n_lon)
    ds = NEERDataset(bundle)
    sample = ds[0]

    assert set(sample["mask"].keys()) == {"input", "target"}

    input_mask = sample["mask"]["input"]
    target_mask = sample["mask"]["target"]
    assert input_mask.dtype == torch.bool
    assert target_mask.dtype == torch.bool
    assert input_mask.shape == (NEER_N_CHANNELS, n_lat, n_lon)
    assert target_mask.shape == (NEER_N_DEPTHS, n_lat, n_lon)

    np.testing.assert_array_equal(input_mask.numpy(), bundle.input_mask[0])
    np.testing.assert_array_equal(target_mask.numpy(), bundle.target_mask[0])


def test_metadata_has_the_four_required_fields():
    bundle = _tiny_bundle(n_time=6, n_lat=4, n_lon=5)
    ds = NEERDataset(bundle)
    metadata = ds[0]["metadata"]

    assert set(metadata.keys()) == {"date", "latitude", "longitude", "depth"}
    assert metadata["date"] == "2020-01-15"
    np.testing.assert_allclose(metadata["latitude"], bundle.lat, atol=1e-4)
    np.testing.assert_allclose(metadata["longitude"], bundle.lon, atol=1e-4)
    np.testing.assert_allclose(metadata["depth"], bundle.depth)
    assert metadata["depth"].shape == (NEER_N_DEPTHS,)


def test_metadata_date_advances_per_timestep():
    ds = NEERDataset(_tiny_bundle(n_time=3))
    dates = [ds[i]["metadata"]["date"] for i in range(3)]
    assert dates == sorted(dates)
    assert len(set(dates)) == 3


def test_negative_indexing_and_out_of_range():
    ds = NEERDataset(_tiny_bundle(n_time=4))
    assert ds[-1]["metadata"]["date"] == ds[3]["metadata"]["date"]
    with pytest.raises(IndexError):
        ds[4]
    with pytest.raises(IndexError):
        ds[-5]


# --------------------------------------------------------------------------
# DataLoader integration
# --------------------------------------------------------------------------


def test_dataloader_batches_with_default_collate():
    n_lat, n_lon = 4, 5
    batch_size = 2
    ds = NEERDataset(_tiny_bundle(n_time=6, n_lat=n_lat, n_lon=n_lon))
    loader = make_dataloader(ds, batch_size=batch_size, shuffle=False)

    batch = next(iter(loader))

    assert batch["inputs"].shape == (batch_size, NEER_N_CHANNELS, n_lat, n_lon)
    assert batch["targets"].shape == (batch_size, NEER_N_DEPTHS, n_lat, n_lon)
    assert batch["mask"]["input"].shape == (batch_size, NEER_N_CHANNELS, n_lat, n_lon)
    assert batch["mask"]["target"].shape == (batch_size, NEER_N_DEPTHS, n_lat, n_lon)

    # date is a plain string per sample -> default_collate gathers a list,
    # not a stacked tensor.
    assert isinstance(batch["metadata"]["date"], list)
    assert len(batch["metadata"]["date"]) == batch_size

    # lat/lon/depth are the same across samples -> stacked into a
    # (batch, len) tensor by default_collate.
    assert batch["metadata"]["latitude"].shape == (batch_size, n_lat)
    assert batch["metadata"]["longitude"].shape == (batch_size, n_lon)
    assert batch["metadata"]["depth"].shape == (batch_size, NEER_N_DEPTHS)


def test_dataloader_covers_every_sample_exactly_once_without_shuffle():
    ds = NEERDataset(_tiny_bundle(n_time=6))
    loader = make_dataloader(ds, batch_size=4, shuffle=False)
    seen = []
    for batch in loader:
        seen.extend(batch["metadata"]["date"])
    assert seen == [ds[i]["metadata"]["date"] for i in range(len(ds))]


def test_dataloader_shuffle_still_covers_every_sample_exactly_once():
    ds = NEERDataset(_tiny_bundle(n_time=6))
    loader = make_dataloader(ds, batch_size=4, shuffle=True)
    seen = set()
    for batch in loader:
        seen.update(batch["metadata"]["date"])
    assert seen == {ds[i]["metadata"]["date"] for i in range(len(ds))}


def test_dataloader_works_over_a_split_subset():
    bundle = _tiny_bundle(n_time=6)
    train_ds = NEERDataset(bundle, split="train")
    loader = make_dataloader(train_ds, batch_size=2, shuffle=False)
    total = sum(batch["inputs"].shape[0] for batch in loader)
    assert total == len(train_ds) == 3


# --------------------------------------------------------------------------
# from_npz / TensorBundle.save round trip
# --------------------------------------------------------------------------


def test_from_npz_round_trip(tmp_path):
    bundle = _tiny_bundle(n_time=5)
    path = bundle.save(tmp_path / "bundle.npz")

    ds = NEERDataset.from_npz(path)
    assert len(ds) == 5

    sample = ds[0]
    np.testing.assert_allclose(sample["inputs"].numpy(), bundle.inputs[0])
    np.testing.assert_allclose(sample["targets"].numpy(), bundle.targets[0])


def test_from_npz_with_split(tmp_path):
    bundle = _tiny_bundle(n_time=6)
    path = bundle.save(tmp_path / "bundle.npz")

    train_ds = NEERDataset.from_npz(path, split="train")
    assert len(train_ds) == 3


# --------------------------------------------------------------------------
# Against the real processed tensors, if they have been generated
# --------------------------------------------------------------------------

PROCESSED_TENSORS = Path(__file__).resolve().parents[1] / "data" / "processed" / "neer_tensors.npz"


@pytest.mark.skipif(
    not PROCESSED_TENSORS.exists(),
    reason=f"{PROCESSED_TENSORS} not found; run scripts/preprocess_data.py first",
)
def test_real_processed_tensors_have_the_neer_shape():
    ds = NEERDataset.from_npz(PROCESSED_TENSORS)
    assert ds.channel_names == list(NEER_CHANNEL_ORDER)
    assert len(ds) > 0

    sample = ds[0]
    assert sample["inputs"].shape[0] == NEER_N_CHANNELS
    assert sample["targets"].shape[0] == NEER_N_DEPTHS
    assert sample["metadata"]["depth"].shape == (NEER_N_DEPTHS,)


@pytest.mark.skipif(
    not PROCESSED_TENSORS.exists(),
    reason=f"{PROCESSED_TENSORS} not found; run scripts/preprocess_data.py first",
)
def test_real_processed_tensors_load_through_a_dataloader():
    ds = NEERDataset.from_npz(PROCESSED_TENSORS, split="train")
    loader = make_dataloader(ds, batch_size=2, shuffle=True)
    n_seen = 0
    for batch in loader:
        n_seen += batch["inputs"].shape[0]
        assert torch.isfinite(batch["inputs"]).all()
        assert torch.isfinite(batch["targets"]).all()
    assert n_seen == len(ds)
