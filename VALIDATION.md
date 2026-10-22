# ARAFNO validation record

Initial validation: October 19, 2026. Environment: Linux, Python 3.12, CPU.

## Initial results

- `pytest -q`: **13 tests passed**. Coverage included hourly windows across
  midnight, NOW interval names, timezone handling, asymmetric crop markers,
  orientation, normalization, missing data, invalid files, incompatible weights,
  artifact integrity, GeoTIFF coordinates, integrated inference, overwrite
  protection, write failures, and simulated FTP caching.
- `ruff check src tests` passed. `uv build` produced a wheel and source
  archive without weights or reference scripts.
- The real checkpoint loaded strictly. The same weights and normalized zero
  input produced a maximum absolute difference of **0.0** against the
  classes in `reference/model.py` for a one-hour prediction.
- The complete pipeline used the real checkpoint with six **synthetic**
  GSMaP binaries and produced six 880 × 970 GeoTIFFs. Reopened outputs
  passed CRS, bounds, times, units, finite nonnegative values, and
  manifest SHA-256 checks.

Synthetic outputs are under:

```text
outputs/validation-synthetic/arafno/gsmap_now/2000/01/01/0500Z/
```

They demonstrate technical behavior, **not real meteorological forecasts**.
Temporary synthetic inputs were removed after testing; their names and hashes
remain in the manifest.

The real JAXA FTP server was not tested because credentials were unavailable.
Listing, caching, and recovery of corrupted files used simulated transport.
Training latitude order and domain bounds are absent from the checkpoint.
The configuration follows the supplied scripts' north-to-south orientation
and bounds. The reference used NRT, while operational inference uses NOW;
skill with NOW remains unverified. GPU/CUDA and the live scheduled service
were not run in this initial validation.

## Run with real NOW files

After the user's FTP run, six files from October 19, 2026, 15:00–20:00 UTC
were inspected. Each frame contained 0.29873%–0.38824% values of −99.
The original `error` policy stopped as designed; the operational setting
was changed to `normalized_zero` with a 5% per-frame threshold.

A local CPU run over that cache published six GeoTIFFs for issue 20:00 UTC
under `outputs/arafno/gsmap_now/2026/10/19/2000Z/`, valid from 21:00
October 19 through 02:00 October 20. The union of missing pixels covered
0.42549% of the domain and was marked NoData. This describes the behavior
**before** the later negative-value correction. Training orientation and
forecast skill with NOW were still unverified.

## Negative values and configuration cleanup

As requested, finite negative values now become zero mm/h before
normalization under every `missing_policy`. NaN and infinities still
follow the missing-data policy. Thus, the earlier −99-as-NoData result above
is a historical record. After this change, **18 tests passed**, including
negative inputs under all three policies and NaN/±infinity. Ruff passed.
Reprocessing the six real cached files produced no missing pixels or output
mask. Previously published GeoTIFFs were not rewritten; use another
`--output-dir` to rerun the same issue.

Later, duplicate configuration defaults were removed: operational JSON
fields no longer have matching dataclass defaults. Native resolution and
global grid live in `formats.py`, with dimensions and byte size derived.
CRS and units use one source in export and manifests. Architecture is
reconstructed from the checkpoint; the reference product comes from the
artifact manifest. **20 tests** and Ruff passed. Real weights and the six
cached observations still produced an unmasked `(6, 880, 970)` input.
Existing forecasts remained unchanged.

## Scheduling and retries — October 20, 2026

- **29 tests passed**, including no-data polls, completed issues, concurrent
  poll protection, pinning the discovered hour, retry after failure,
  stale/future data, and integrity of published outputs.
- Simulated transfers covered network errors, incomplete gzip grids,
  exponential delays, exhausted retries, and avoidance of invalid cache files.
- Ruff, `systemd-analyze verify`, and `systemd-analyze calendar` passed.
  Transport parameters have one source in JSON `ftp`; poll frequency has
  one source in the `.timer`.

No service was activated: the user bus returned `Operation not permitted`
in this session. Continuous polling against real FTP was not performed.
See [deploy/README.md](deploy/README.md) for installation and credentials.

## Runtime distribution — October 21, 2026

Acquisition, processing, GeoTIFF export, publication, and scheduling were
first extracted into `shared/nowcasting_runtime`; ARAFNO's 29 tests passed
again. Units were regenerated and verified without activation.

The final independent distribution embeds its private runtime under
`src/arafno/_runtime/`. `uv lock --offline` and `uv build --offline`
passed at that stage. The wheel contains 11 private runtime modules and has
no dependency on `nowcasting-runtime` or `../shared`; importing from an
extracted wheel outside the checkout also passed. The 29 tests passed again
with `uv run --offline --locked --group dev pytest -q`. Runtime changes
needed by multiple independent models must be applied to each model.

## Temporary download retention — October 21, 2026

The default FTP cache moved to the system temporary directory, isolated by
user and model. `--cache-ttl-hours` controls retention (24 hours by
default). Cleanup removes only expired files in managed namespaces; it does
not touch local inputs or unrelated directories. It also runs when a poll
finds an already-completed issue.

**31 tests passed** with
`uv run --offline --locked --group dev pytest -q`; Ruff and
`uv lock --offline --check` passed. The five-minute timer expression passed
`systemd-analyze calendar`. Rebuilding the wheel in that offline session
was blocked because `hatchling`, required by the build, was absent from the
cache. The previous build had passed; the current source changes were
validated by tests.
