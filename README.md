# NEER

**N**eural **E**mbedding based **E**stimation and **R**econstruction

- **SIH Problem Statement:** SIH26066
- **Organization:** MoES / INCOIS

> ⚠️ **Phase 12 — PyTorch Dataset.** The data layer is complete through
> tensor assembly, and `src/data/dataset.py` now wraps the assembled
> tensors in a `torch.utils.data.Dataset`/`DataLoader` pair. **The ML
> model itself is not implemented yet** — nothing in `src/models/`,
> `src/training/` or `src/evaluation/` is built.

## Project structure

```
NEER/
├── backend/            # FastAPI backend (minimal: /health only)
├── frontend/            # Next.js frontend (minimal landing page)
├── src/                 # Core ML/data-science codebase
│   ├── data/             # Loaders, validation, preprocessing pipeline
│   ├── models/           # Model architectures (not implemented yet)
│   ├── training/         # Training loops / experiment orchestration
│   ├── evaluation/       # Metrics & validation
│   ├── explainability/   # Interpretability tooling
│   └── utils/            # Shared utilities (config, logging, ...)
├── configs/              # YAML configuration files
├── data/
│   ├── raw/               # Raw, unprocessed data
│   ├── processed/         # Cleaned/processed data
│   └── demo/               # Small demo datasets
├── artifacts/
│   ├── checkpoints/        # Model checkpoints
│   ├── embeddings/         # Saved embeddings
│   └── predictions/        # Model predictions/outputs
├── reports/              # Generated reports/analyses
├── scripts/              # Utility & operational scripts
├── tests/                # Test suite
├── requirements.txt
├── README.md
├── MODEL_CARD.md
├── .gitignore
├── Dockerfile
└── docker-compose.yml
```

## Quickstart

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check it:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "project": "NEER",
  "problem_id": "SIH26066"
}
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`.

### Docker Compose (both services)

```bash
docker-compose up --build
```

- Backend → http://localhost:8000
- Frontend → http://localhost:3000

## Data loading

`src/data/loaders/` turns NetCDF, CSV, and NumPy-compatible sources into
one standardized internal representation, `OceanDataset`:

```python
from src.data.loaders import load_demo_dataset, load_netcdf, load_csv

dataset = load_demo_dataset()              # synthetic demo data (data/demo)
dataset = load_netcdf("argo_grid.nc")      # gridded products
dataset = load_csv("profiles.csv")         # point/profile observations

dataset["sst"].shape                       # (24, 101, 241)
dataset["sst"].dims                        # ('time', 'lat', 'lon')
dataset["sst"].units                       # 'degC'
print(dataset.describe())
```

Whatever the source format, the result carries the same guarantees:

| Guarantee | What it means |
|---|---|
| Canonical coordinates | Always `time`, `depth`, `lat`, `lon` — never `latitude`, `nav_lat`, `lev`, `t` |
| Canonical axis order | Every variable is `(time, depth, lat, lon)`, restricted to the dims it has |
| Decoded timestamps | `time` is `datetime64[ns]`, strictly increasing |
| Missing data is NaN | `-999`, `1e20`, `_FillValue` and friends are resolved at load time |
| Canonical units | `degC`, `psu`, `m`, `m/s`; Kelvin/cm/knots are converted, with the original kept in `attrs["source_units"]` |
| Preserved provenance | `data_mode` and disclaimers survive the load — `dataset.is_synthetic` flags demo data |

Every dataset is validated on load (coordinates, timestamps, variables,
dimensions, units, missing values). Errors — a descending latitude axis, a
duplicated timestamp, a fully-missing variable — raise
`DataValidationError`. Warnings — an unknown variable, an undeclared unit,
values outside a plausible range — are recorded on
`dataset.validation` and can be made fatal with `strict=True`:

```python
dataset = load_netcdf("suspicious.nc", strict=True)
print(dataset.validation)            # full report
dataset.validation.to_dict()         # JSON, for logs and API responses
```

xarray and netCDF4 are imported lazily, so only `load_netcdf` needs them;
the CSV, NumPy, and demo paths work without them installed.

⚠️ `load_demo_dataset()` returns **synthetic** data (`data_mode =
"DEMO_SYNTHETIC"`). Anything that reports results to a user should check
`dataset.is_synthetic` and label the output accordingly.

## Checkpoint management (Phase 27)

`src/training/checkpoint.py` is the one place every training script saves and
loads checkpoints. `scripts/train.py` uses it for `neer_best.pt` / `neer_last.pt`
and `--resume`.

```python
from src.training.checkpoint import CheckpointManager

mgr = CheckpointManager("artifacts/checkpoints", name="neer")
mgr.save_last(model=model, optimizer=optimizer, epoch=epoch, seed=seed, config=config,
              training_stats={"val_loss": val_loss, "history": history})
if improved:
    mgr.save_best(model=model, optimizer=optimizer, epoch=epoch, seed=seed,
                  config=config, training_stats={"val_loss": val_loss})

loaded = mgr.load_checkpoint(mgr.best_path, model=model, optimizer=optimizer, config=config)
print(loaded.epoch, loaded.training_stats, loaded.report.warnings)
```

Each checkpoint bundles model + optimizer state with the epoch, seed, the
resolved config (plus a fast `config_fingerprint` of its architecture-relevant
fields), training statistics, parameter shapes, and git/project metadata
(commit, branch, dirty flag — degrades to `{"available": false}` outside a repo
or without git installed, never an error). A human-readable `.json` sidecar is
written next to every `.pt` file. Saves are atomic (write-to-temp then rename),
so a crash mid-write never corrupts the previous checkpoint.

**Compatibility is checked before any weights are touched.** Loading diffs the
checkpoint's stored parameter shapes against the live model: a shape mismatch is
always a hard error (`CheckpointCompatibilityError`, raised before
`load_state_dict` runs); missing/unexpected parameter names are errors under
`strict=True` (the default) and warnings under `strict=False`. A config
fingerprint mismatch is only a warning — the shape check is authoritative.
Checkpoints saved before this module existed still load, as a `"legacy"`
schema with best-effort compatibility checking.

## ARGO validation (Phase 26)

`src/argo_validation/` validates NEER's predicted temperature profiles against ARGO
float profiles. It is **separate from** the reanalysis/test-target evaluation in
`src/evaluation` (different data, matching and outputs; only the four scalar metric
functions are shared) and writes to `reports/argo_validation/`.

```
ARGO profiles -> QC -> time matching -> spatial matching
  -> vertical interpolation -> NEER prediction -> comparison -> metrics
```

Reported: float count, profile count, matched profiles, depth coverage, RMSE, MAE,
bias (NEER - ARGO) and correlation — overall, per depth and per NEER split.

```bash
# real ARGO data (long-format CSV, GDAC *_prof.nc, or a directory of .nc files)
python scripts/run_argo_validation.py --argo data/raw/argo_profiles.csv \
    --neer-checkpoint artifacts/checkpoints/neer_best.pt
python scripts/run_argo_validation.py --argo data/raw/argo_nc/ --predictions preds.npz

# no real ARGO data: a clearly labelled SYNTHETIC demo (pipeline check only)
python scripts/run_argo_validation.py --demo
```

⚠️ **Demo ARGO data is synthetic and never observational validation.** `--demo` is the
only way demo profiles are created; every output is prefixed `DEMO_SYNTHETIC_`, the
report has `observational_validation: false` and files its numbers under
`pipeline_check_metrics` (not `metrics`). Without a checkpoint, demo runs use a stand-in
predictor labelled `demo_stand_in_NOT_NEER`.

Read before interpreting real results: NEER currently outputs one **domain-pooled**
profile per month, so each ARGO profile is compared to the month's basin-mean profile
(the error includes spatial spread); and profiles in NEER's *train* months may not be
independent of the model if its targets derive from ARGO — use the `test`/`val` rows of
`by_split`.

## Data validation

`src/data/validation/` audits a loaded dataset against the project
configuration and against physical reality, and returns a report with a
three-level status — **valid / warning / error**:

```python
from src.data.loaders import load_demo_dataset
from src.data.validation import validate, SURFACE_VARIABLES

report = validate(load_demo_dataset(), required_variables=SURFACE_VARIABLES)
report.status              # ValidationStatus.VALID
report.is_usable           # True when there are no errors
print(report.to_text())
report.save("reports/validation.md")   # .json / .md / .txt
```

Ten checks run by default:

| Check | Error | Warning |
|---|---|---|
| `latitude_range` | outside ±90°, descending, NaN | outside the configured domain |
| `longitude_range` | outside ±360°, descending, NaN | outside the configured domain |
| `resolution` | uneven grid spacing | uniform but ≠ `domain.resolution` |
| `depth_coordinates` | negative, unsorted, implausibly deep | ≠ the configured depth levels |
| `time_coordinates` | undecoded, NaT, unsorted, wrong epoch | irregular cadence |
| `variable_existence` | a required variable absent | variable not in the registry |
| `units` | unit contradicts the variable | undeclared or unrecognized |
| `nan_values` | 100% missing, infinities | >95% missing, implausible values |
| `duplicate_timestamps` | any repeated timestamp | — |
| `duplicate_coordinates` | any repeated lat/lon/depth | — |

The severity split is the point: a latitude of 130° is impossible and
blocks the pipeline, while 45°N is real data that merely needs subsetting.

### Nothing is repaired

Findings carry a suggested `remedy` in prose, for a human to act on.
**This layer never modifies the data it validates** — a serious problem is
reported, never silently worked around:

```python
report.remedies()   # ['sort the dataset along lat ...', ...]
```

`validate()` never raises; use `validate_or_raise(dataset, strict=True)`
to enforce, or `DataValidator(...)` to apply one contract across many
files.

### From the command line

```bash
python scripts/validate_data.py                              # the demo dataset
python scripts/validate_data.py data/raw/argo.nc --require-surface
python scripts/validate_data.py --report reports/validation.md --strict
```

Exit codes: `0` valid, `1` errors (or warnings with `--strict`), `2` the
file could not be loaded.

This is the scientific audit of a dataset that already loaded. The
separate load-time structural gate in `src/data/loaders/validation.py`
decides whether a file can become an `OceanDataset` in the first place.

## Preprocessing

`src/data/preprocessing` turns a loaded `OceanDataset` into the tensors a
model consumes:

```
raw data
  -> coordinate normalization   one longitude convention, ascending axes
  -> temporal alignment         one regular cadence, gaps made explicit
  -> spatial alignment          resampled onto the configs/base.yaml grid
  -> missing-value handling     LEARNED: climatology fitted on train only
  -> masking                    land blanked, observation masks published
  -> feature construction       speeds, cyclic time, position, Coriolis
  -> normalization              LEARNED: statistics fitted on train only
  -> tensor assembly            (time, channel, lat, lon) + masks + split
```

Run it:

```bash
python scripts/preprocess_data.py                      # the demo dataset
python scripts/preprocess_data.py data/raw/argo.nc -o data/processed/argo
python scripts/preprocess_data.py --dry-run            # report, write nothing
```

Or from Python:

```python
from src.data.loaders import load_demo_dataset
from src.data.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline.from_config()
result = pipeline.run(load_demo_dataset())

result.tensors.inputs.shape        # (24, 14, 101, 241)  time, channel, lat, lon
result.tensors.targets.shape       # (24, 15, 101, 241)  time, depth, lat, lon
result.train.n_time                # 16
result.save("data/processed")      # tensors + metadata + readable report
```

Every step is a standalone, reusable object with the same contract —
`fit`, `transform`, `config()`, `state()`, `report()` — so a single stage
can be used on its own:

```python
from src.data.preprocessing import CoordinateNormalizer
dataset = CoordinateNormalizer(lon_convention="-180-180").transform(dataset)
```

### No test data reaches training

Two rules, both enforced in code rather than by convention:

1. **Steps that learn are fitted on the training period only.** Each step
   declares `learns_from_data`; before any `fit`, the pipeline calls
   `assert_fit_window` on the timestamps actually present in the slice it
   is about to fit on, and raises `LeakageError` if any fall outside the
   training period.
2. **After the split, each period is transformed separately.** Fitting on
   training data is not sufficient on its own: `MissingValueHandler`
   bridges short gaps along the time axis, so a hole in the last training
   month would otherwise be filled from the first validation month — no
   parameter fitted, no guard tripped, validation data inside the
   training tensor. Train, validation and test never share an array after
   the split, so no step can reach across a boundary.

The split is always chronological (train, then validation, then test),
never random: ocean fields are strongly autocorrelated in time, so a
random split puts near-duplicate timesteps on both sides of the boundary
and every score comes out optimistic for the wrong reason.

`tests/test_preprocessing_leakage.py` tests this the only way that
convinces: it corrupts every held-out timestep beyond recognition,
re-runs, and requires that not one number in the training tensors moved.

### Masks are the contract

`inputs` and `targets` are guaranteed finite — a model cannot consume
NaN, so every remaining gap is filled with `fill_value` (0.0, the
training mean after normalization). The *only* record of what was real is
the mask, which is True exactly where a cell is ocean, finite, and
actually measured rather than gap-filled.

That makes `target_mask` load-bearing. Subsurface coverage is sparse —
about 24% of target cells in the demo run — and a loss computed without
the mask would train the model to reproduce fill values over land and
score it on gaps interpolated from its own inputs.

### Saved metadata

Every run writes `neer_preprocessing_metadata.json` beside the tensors,
recording each step's configuration, fitted state and per-run report; the
split and the exact fit window each learned step saw; the tensor shapes
and channel order; and the provenance, so demo tensors keep their
`DEMO_SYNTHETIC` marker and can never be mistaken for real results.

The normalization statistics are stored in full, deliberately — they are
what a later phase needs to put predictions back into degrees Celsius:

```python
from src.data.preprocessing import PreprocessingMetadata
metadata = PreprocessingMetadata.load("data/processed/neer_preprocessing_metadata.json")
metadata.normalization_statistics()["subsurface_temp"]   # center/scale per depth level
```

## Monthly climatology (delta_T)

`src/data/preprocessing/climatology.py` fits a monthly temperature
climatology — the training-period mean for each cell in each calendar
month — and exposes it as a small, reusable, standalone API (it is not
one of the eight default pipeline stages, and can be fit and queried on
its own, like any other preprocessing step):

```python
from src.data.preprocessing import MonthlyClimatology

climatology = MonthlyClimatology().fit(train_dataset)   # training split only

climatology.monthly_field("sst", month=7)                        # monthly lookup
climatology.at("subsurface_temp", month=7, lat=12.0, lon=80.0, depth=100)  # spatial + depth lookup
climatology.depth_profile("subsurface_temp", month=7, lat=12.0, lon=80.0)  # depth lookup
```

It is built around one exactly-invertible relationship, used to turn a
temperature field into a training target and a predicted anomaly back
into a temperature:

```python
from src.data.preprocessing import compute_delta_temperature, reconstruct_temperature

delta_T = compute_delta_temperature(target_temperature, climatology_field)
temperature_prediction = reconstruct_temperature(climatology_field, predicted_delta_T)
```

`MonthlyClimatology.delta_for`/`.reconstruct` do the same thing matched
against a dataset's own `time` axis, `.transform()` attaches
`<var>_climatology`/`<var>_delta_t` variables to a dataset, and
`.save`/`.load` persist a fitted climatology to a `.npz` file so a later
phase can reconstruct predictions without re-fitting. See
`tests/test_climatology.py`.

## PyTorch Dataset (Phase 12)

`src/data/dataset.py` wraps a `TensorBundle` (Phase 08) in
`NEERDataset`, a `torch.utils.data.Dataset` where one sample is one
timestep's full `(channel, lat, lon)` grid:

```python
from src.data.dataset import NEERDataset, make_dataloader
from src.data.preprocessing.tensors import TensorBundle

bundle = TensorBundle.load("data/processed/neer_tensors.npz")
train_ds = NEERDataset(bundle, split="train")   # or NEERDataset.from_npz(path, split="train")
loader = make_dataloader(train_ds, batch_size=4, shuffle=True)

batch = next(iter(loader))
batch["inputs"]             # (4, 11, lat, lon) float32 — NEER_CHANNEL_ORDER
batch["targets"]            # (4, 15, lat, lon) float32 — subsurface temperature per depth
batch["mask"]["input"]      # (4, 11, lat, lon) bool
batch["mask"]["target"]     # (4, 15, lat, lon) bool — mask any loss with this
batch["metadata"]["date"]       # list[str], length 4
batch["metadata"]["latitude"]   # (4, lat)
batch["metadata"]["longitude"]  # (4, lon)
batch["metadata"]["depth"]      # (4, 15) metres
```

torch is an optional dependency (like `xarray`/`netCDF4` for the NetCDF
loader): importing `src/data/dataset.py` works without it installed;
constructing `NEERDataset` without it raises `MissingDependencyError`
naming `pip install torch`. See `tests/test_dataset.py`.

## Optional graph refinement (Phase 22)

`src/models/gnn.py` adds an *optional* graph neural network that refines
the CNN's per-cell features before the ViT. Grid cells are the nodes;
neighbouring cells (4-connected by default, 8 optionally) are the edges.

```
(batch, 11, lat, lon) -> CNNEncoder -> [GridGNN, if use_gnn] -> ViT -> decoder -> (batch, 15)
```

It is off by default — `model.use_gnn: false` in `configs/base.yaml` —
and when off no GNN module is built at all, so the model, its
parameters and its `state_dict` are exactly what they were without it.
Turn it on in the config, or per run with `NEER_MODEL_USE_GNN=true`:

```python
from src.models import GNNConfig, NEERModel
from src.utils.config import load_config

model = NEERModel.from_config(load_config("demo"))        # follows model.use_gnn
model = NEERModel(use_gnn=True)                           # or explicitly
model = NEERModel(use_gnn=True, gnn_config=GNNConfig(num_layers=2, connectivity=8))
```

With `GNNConfig.use_current=True` (default) the model's own
`u_current`/`v_current` input channels are projected onto every edge
(flow along and across it) and fed into the message function, so the
ocean current changes what each cell hears from each neighbour. With
`use_current=False` the current is not read at all. The GNN adds about
6.5k parameters to the default model, and pretrained CNN/ViT weights
(`load_pretrained_encoder`) load unchanged either way. Not modelled: land
masking (the input has no mask channel) and the east-west/north-south
cell-size difference away from the equator. See `tests/test_gnn.py` and
`tests/test_gnn_graph.py` (the latter needs no torch).

## Configuration

Base configuration lives in `configs/base.yaml` and is loaded via
`src/utils/config.py`. Values can be overridden with `NEER_*` environment
variables (e.g. `NEER_ENVIRONMENT=production`).

The `preprocessing:` section declares the pipeline — cadence, target
grid, fill strategies, features, normalization method and split
fractions — and is consumed by `PreprocessingPipeline.from_config()`.
Every key is optional; anything omitted falls back to the step's own
default.

## Status

Phases 01-12 are complete: project scaffold, config system, ocean grid,
data loaders, validation, the preprocessing pipeline, channel/tensor
assembly, and the PyTorch `Dataset`/`DataLoader` layer. Raw data goes in;
normalized tensors, metadata, and now batched, model-ready `torch`
batches come out.

Not yet built: the model architecture (`src/models/`), the training loop
(`src/training/`), evaluation metrics (`src/evaluation/`), explainability
(`src/explainability/`), and the dashboard UI beyond its landing page.
See `MODEL_CARD.md` for the outstanding work.