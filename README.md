# ARAFNO — operational GSMaP NOW inference

This independent pipeline acquires GSMaP inputs, reconstructs the ARAFNO model,
loads weights strictly, preprocesses observations, and publishes autoregressive
forecasts as GeoTIFFs. It uses six consecutive hourly **GSMaP NOW** inputs and
publishes six hourly leads by default. It does not require future observations,
Zarr, Cartopy, or matplotlib.

Acquisition, processing, export, and scheduling are packaged in
`src/arafno/_runtime/`. This project runs without the other model directories
or `../shared/nowcasting_runtime`. After updating an existing installation,
run `uv sync --locked`.

## Project layout

```text
01_ARAFNO/
├── pyproject.toml / uv.lock     # installable package and reproducible dependencies
├── config/latam.json           # domain, product, orientation, missing data, artifacts
├── src/arafno/
│   ├── model.py                # architecture, weight loading, autoregressive rollout
│   ├── _runtime/               # private acquisition, processing, GeoTIFF, scheduler
│   ├── acquisition.py          # local files, FTP, cache, time-window selection
│   ├── processing.py           # binary reading, cropping, normalization
│   ├── export.py               # GeoTIFF writer
│   ├── artifacts.py            # model bundle integrity
│   ├── config.py               # configuration contract
│   ├── formats.py              # GSMaP format and output constants
│   ├── scheduler.py            # scheduled polling, locking, deduplication
│   └── entrypoint.py           # CLI orchestration
├── artifacts/v1/               # weights, mean, std, SHA-256 manifest
├── reference/                  # optional local reference scripts; ignored by Git
├── tests/                      # temporal, spatial, and integration checks
├── deploy/                     # systemd timer, generator, installation guide
├── data/raw/                   # optional local inputs; never deleted by pipeline
└── outputs/                    # generated forecasts, excluded from Git
```

## Configuration and installation

Each operational parameter has one source of truth:

| Parameters | Source |
|---|---|
| Domain bounds, product, remote directory, orientation, artifacts, missing-data policy | Required fields in `config/latam.json` |
| Native resolution, global binary shape, CRS, units | `src/arafno/formats.py`; dimensions and byte size are derived |
| Input channels, embedding, depth, patch size | Checkpoint tensors; no alternative constructor defaults |
| Input source, directories, horizon, device, threads | CLI arguments and parser defaults |
| Credentials | Environment variables |
| FTP timeout and retry policy | `ftp` section of `config/latam.json` |
| Poll frequency | `OnCalendar` in `deploy/arafno-now.timer` |

The 0.1° resolution is part of the GSMaP binary contract, not a resampling
option. Older JSON configurations must remove `resolution`. The effective
resolution is recorded in each run manifest. Architecture values in the
artifact manifest are descriptive; reconstruction reads the checkpoint.
Local scripts under `reference/` can be retained for auditing; Git ignores
this directory, and it is not required to install or run the operational
pipeline. Their comments and display text were translated to English;
model calculations were not changed.

Use Python 3.11 or newer. From this directory:

```bash
uv sync --locked --group dev
uv run --locked arafno --help
```

The lock targets CPU PyTorch for a portable installation. For CUDA, install a
PyTorch build compatible with the driver and then `pip install .` in that
environment. Run `arafno --device cuda ...` there. Do not synchronize the
CPU lock over a CUDA installation.

## Acquire inputs and run inference

For original global GSMaP binaries stored locally, in mm/h:

```bash
uv run --locked arafno --config config/latam.json \
  --source local --input-dir data/raw/now \
  --run-time 2026-10-19T12:00:00Z --steps 6 --device cpu
```

That example requires `gsmap_now.20261019.0700.dat.gz` through
`gsmap_now.20261019.1200.dat.gz`. Without `--run-time`, the latest
observation is used only if its entire input window is present. Missing hours
do not silently shift the run backward. Local directories are searched
recursively; duplicate hours are rejected.

The reader also accepts NOW names with explicit intervals, such as
`gsmap_now.20261019.1200_1300.dat.gz`; the initial hour identifies the
observation. Half-hour (`HH30`) files are outside the hourly sequence.

For operational FTP access:

```bash
export FTP_JAXA_HOST=hokusai.eorc.jaxa.jp
export FTP_JAXA_USER='your_username'
export FTP_JAXA_PASSWORD='your_password'
uv run --locked arafno --config config/latam.json --source ftp \
  --output-dir outputs --device auto
```

By default, downloads live under
`<system-temp>/nowcasting-<UID>/arafno` and expire after 24 hours. Set
`--cache-ttl-hours 6` to change retention or `--cache-dir` for another
managed cache. Cleanup runs on each FTP invocation or scheduled poll, even
when the forecast already exists. With isolated manual runs, expired files
remain until the next run or operating-system cleanup. The pipeline never
deletes files supplied through `--source local --input-dir`.

Credentials are read only in FTP mode and are never stored in manifests.
Keep them out of the repository. FTP requests have a timeout and up to three
attempts. A download is held temporarily and its gzip is validated before it
enters the cache; invalid cached files are fetched again. Without an explicit
run time, FTP mode rejects future observations or those older than
`--max-age-hours` (default: 12 hours).

To reprocess historical files, configure `remote_directory` with the
provider's archive path. Templates such as `%Y/%m/%d` expand across every
day in the input window and require `--run-time`. `/now/latest` only contains
recent files; an old timestamp does not make the archive available. Paths in
the JSON are relative to that file; CLI paths are relative to the working
directory.

For continuous operation, follow the [systemd scheduling guide](deploy/README.md).
`--source ftp --scheduled` performs one idempotent poll. The timer checks every
five minutes and runs inference only for a new hourly issue. It requires a
complete window, reuses cached data, and retries transport failures and
incomplete uploads. Units must be installed and enabled on the operational
host; generating them does not activate the service.

## Scientific and spatial contract

- Verified checkpoint shape: input `[1, 6, 880, 970]`, embedding 256,
  depth 8, 10 × 10 patches, one-hour model output. Dimensions are
  reconstructed from the checkpoint, checked against the configured domain,
  and loaded with `strict=True` and `weights_only=True`.
- The reader uses little-endian float32 in a global 1200 × 3600 grid,
  converts longitudes from 0–360° to −180–180°, and crops
  `[-120, -55, -23, 33]`. The resulting 0.1° grid starts at pixel center
  `(-119.95, 32.95)` with north at the top.
- Normalization is `(log1p(rain) - mean) / std`, using the supplied
  `.npy` files. Output uses `max(expm1(prediction * std + mean), 0)`.
  The next autoregressive step receives the normalized prediction, not
  denormalized or clipped rainfall.
- GeoTIFF values are **precipitation rates in mm/h**, not accumulations.
  `run_time` is the last input timestamp and
  `valid_time = run_time + lead_hours`. Timestamps retain the source
  product's convention.
- Finite negative inputs, including −99, become **zero mm/h before
  normalization**. They are not counted as missing and do not create NoData.
  Negative values in normalized space remain valid.
- NaN and infinite inputs are missing. The operational setting
  `missing_policy: "normalized_zero"` fills missing values with zero in
  normalized space, matching the reference's NaN treatment; that is not
  equivalent to zero rainfall. `error` stops on any missing input;
  `zero_rain` fills zero mm/h before normalization. With a filling policy,
  pixels missing in any input receive NoData in every forecast. Each frame
  must also satisfy `max_missing_fraction` (default: 5%). This mask does
  not quantify uncertainty propagated by the model's spatial filters.

**Scientific validation remains open.** The reference inference used NRT,
whereas this operational pipeline uses NOW. Forecast skill with NOW needs
evaluation. The checkpoint does not record training latitude order.
`model_latitude_order: "north_to_south"` follows the operational reference
but must be checked against georeferenced training data. If training used
south-to-north order, set `south_to_north`; the pipeline flips inputs and
returns outputs to north-up order. Domain bounds come from the supplied
scripts; weights confirm grid size only. Increasing `--steps` does not
establish skill beyond the evaluated horizon.

Provider format references:
[GSMaP NOW](https://sharaku.eorc.jaxa.jp/GSMaP/document/DataFormatDescription_NOW.pdf)
and [GSMaP NRT](https://sharaku.eorc.jaxa.jp/GSMaP/document/new/DataFormatDescription_NRT8.pdf).

## Output and provenance

```text
outputs/arafno/gsmap_now/2026/10/19/1200Z/
├── arafno_20261019T1200Z_f001_20261019T1300Z.tif
├── ...
├── arafno_20261019T1200Z_f006_20261019T1800Z.tif
└── manifest.json
```

Each file has one float32 band, EPSG:4326, area pixels, NoData −9999,
256 × 256 tiling, and DEFLATE compression. Tags include the model, version,
checkpoint hash, product, issue, valid time, lead, and unit. The manifest also
records input and output hashes, configuration, normalization, missing-data
information, and library versions. A run directory is renamed into place
only after every output is complete. Failed runs do not publish partial
forecasts; existing runs are never overwritten. Use another `--output-dir`
for a different model or configuration.

## Distribution and verification

Separating the model, acquisition, and entrypoint is appropriate. Processing
and export have their own modules because their scientific and spatial
contracts must be checked independently of transport and weights.

`01_ARAFNO` can remain a human-readable catalog directory. The importable
package is `arafno`; its wheel includes the private runtime and model code.
For distribution, ship the **wheel, versioned artifact bundle, and
configuration** together, keeping weights and statistics paired and checking
their manifest. SHA-256 detects changes but does not establish provenance.

The 93 MB checkpoint belongs in a release, object storage, or Git LFS; it is
ignored by ordinary Git. The `.npy` files and manifest may be versioned.
Before public distribution, establish the license, weight provenance, and
applicable data redistribution terms. Local reference assets are for auditing,
are excluded from this repository, and are not maintained entrypoints.

One GeoTIFF per issue and lead simplifies GIS use without merging forecasts
with the same valid time but different issues. Large remote services could
later add COG and STAC; these are ordinary tiled GeoTIFFs, not certified COGs.
Zarr remains useful for large scientific series with separate issue and lead
axes. Raw cache, model artifacts, and derived products have different life
cycles and should not share one flat directory.

```bash
uv run --locked pytest
uv run --locked ruff check src tests
uv build
```

Tests use identifiable synthetic data and small models to check time,
orientation, normalization, simulated acquisition, strict loading, and
GeoTIFF export. They do not replace meteorological validation with NOW.
See [VALIDATION.md](VALIDATION.md) for recorded results and synthetic output
locations.
