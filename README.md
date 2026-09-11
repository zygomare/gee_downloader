# EO Downloader

Download satellite images as GeoTIFFs for a given AOI (shapefile) and date range.
Supports four backends selectable via `GLOBAL.backend` in the config file.

## Backends

| Backend | Key | Satellites | Notes |
|---------|-----|------------|-------|
| Google Earth Engine | `gee` | S2, S1, Landsat 8/9, AlphaEarth | Tiles AOI into sub-cells to stay within GEE pixel limits |
| Google Cloud Storage | `gcld` | S2 L1 (SAFE) | Downloads via `gsutil`, generates TOA with ACOLITE. **Only Sentinel-2 is actually implemented** — see [Known limitations](#known-limitations--inert-config) |
| STAC API | `stac` | S2 L1/L2, Landsat 8 | Uses a vendored downloader with a hardcoded endpoint list and collection map (Element84 Earth Search, then Planetary Computer) |
| Copernicus Data Space | `cdse` | S3 OLCI L1 | Downloads via CDSE OData API, generates TOA with ACOLITE; cloud masking via IdePix or native OLCI flags |

## Installation

```bash
pip install -r requirements.txt
```

**GEE / GCLD backends** — also install and authenticate the gcloud CLI:
```bash
gcloud auth login
gcloud auth application-default login
```
Register your Google Cloud project for Earth Engine: https://code.earthengine.google.com/register

**GCLD / CDSE backends** — require [ACOLITE](https://github.com/acolite/acolite) for TOA reflectance generation. Set `acolite_dir` in the config to your local ACOLITE clone.

**CDSE backend (cloud masking)** — controlled by `cloud_mask_method` in the `CDSE` config section:

| Value | Behaviour |
|-------|-----------|
| `idepix` (default) | ESA SNAP IdePix; auto-falls back to native flags on pre-AVX2 CPUs or if SNAP/`esa_snappy` is unavailable |
| `native` | Native OLCI `qualityFlags.nc` bits (CLOUD, CLOUD_AMBIGUOUS, CLOUD_SHADOW); no SNAP required |

To enable IdePix:
1. Install [ESA SNAP](https://step.esa.int/main/download/snap-download/)
2. `pip install esa_snappy`
3. `<SNAP_dir>/bin/snappy-conf <python_executable>`
4. Install the IdePix plugin: `<SNAP_dir>/bin/snap --modules --install org.esa.snap.idepix.core org.esa.snap.idepix.olci --nogui --nosplash`

**PROJ conflicts** — if another application (e.g. SeaDAS) sets `PROJ_LIB`/`PROJ_DATA` in the environment, set `proj_data` in the `GLOBAL` config section to override it before any geo library is loaded:
```yaml
GLOBAL:
  proj_data: /path/to/proj/data   # directory containing proj.db
```
To find the right path: `python -c "import pyproj; print(pyproj.datadir.get_data_dir())"`. See `docs/troubleshooting.md` for details.

## Usage

```bash
python main.py -c download.yaml
```

Edit `download.yaml` to set the backend, AOI path, date range, assets, and output directory. A timestamped copy of the config is saved to `save_dir` on each run.

---

## How the config file is parsed

`download.yaml` has up to five top-level blocks: `GLOBAL`, one backend-named block (`GEE`, `GCLD`, `STAC`, or `CDSE`) matching whatever `GLOBAL.backend` is set to, and `ASSETS`.

The loader (`utils.load_config_file` → `convert_yaml_to_internal_config`, `utils.py:344-486`) flattens this at load time:

- All keys in `GLOBAL:` are lowercased and become `config['global'][...]`.
- The backend-named block (e.g. `GCLD:` when `backend: gcld`) is merged into that **same** `config['global']` dict — but **only for keys not already present in `GLOBAL:`**. In practice this means there is no real section isolation at runtime: a key like `acolite_dir` under `GCLD:` and `project_id` under `GEE:` both just become flat entries on `config['global']`, and every downloader class reads them the same way, off `self._config_dic['global']`.
- The other backend-named blocks you're *not* using are simply never merged in and have no effect (e.g. leaving `CDSE:` populated while `backend: gee` is harmless).
- Each `ASSETS:` list entry (`- ASSET_NAME: {...}`) becomes its own section, `config['<asset_name_lower>']`. The per-asset `<backend>_source` key matching the active backend (e.g. `gee_source` when `backend: gee`) is renamed to a generic `source` field; the *other* `*_source` keys for backends you're not using are dropped, except `gcld_source`/`cdse_source`, which currently pass through unused (see [Known limitations](#known-limitations--inert-config)).
- A value of `NONE` (any case) for a `*_source` key becomes Python `None`.
- `include_bands` can be written as a YAML list or a comma string; both are normalized to a comma string.
- Booleans written unquoted (`true`/`false`) are parsed by PyYAML as real Python `bool`s — **most of the time this is fine**, but one option (`obs_geo_pixel`) is compared against the strings `'True'`/`'true'` in code, so an unquoted YAML boolean silently fails there. See [Known limitations](#known-limitations--inert-config).

`main.py` dispatches to the downloader class from `GLOBAL.backend` (case-insensitive; `gee` if unset).

---

## Configuration reference

### `GLOBAL` (read by the base `Downloader` class — applies to all backends)

| Key | Default | Required | Purpose |
|---|---|---|---|
| `backend` | `gee` | — | Selects the downloader: `gee`, `gcld`, `stac`, or `cdse`. |
| `aoi` | — | **yes** | Path to the AOI vector file (shapefile/geojson, CRS EPSG:4326), loaded with geopandas. Its filename (minus extension) becomes the `project_name` segment appended to `save_dir`. If the AOI has a `name` column, each feature's `name` value is used as the per-cell output folder name; otherwise a generated index is used. |
| `save_dir` | — | **yes** | Root output directory. The actual output root used is `<save_dir>/<aoi_file_stem>/`, created if missing. |
| `assets` | — | **yes** | Comma-separated list of `ASSETS:` entry names to process (e.g. `S2_L1TOA,S2_L2RGB`). An asset name with no matching `ASSETS:` entry just logs a warning and is skipped — not fatal. |
| `start_date` / `end_date` | — | required unless `date_csv` is set | **Single range mode**: full `YYYY-MM-DD` dates, one continuous window. |
| `start_year` / `end_year` + `start_date`/`end_date` | — | optional | **Multi-year seasonal mode**: `start_date`/`end_date` become `MM-DD` (no year), and the same seasonal window is repeated for every year in `[start_year, end_year]`. Cross-year windows (e.g. `12-01` → `02-28`) are handled correctly (end date rolls into `year+1`). |
| `date_csv` | none | optional (mandatory in practice for GCLD) | CSV overriding date-range logic per AOI cell. Must have `name` and `date` (YYYYMMDD) columns at minimum; both are cast to string. See per-backend notes below — GCLD and CDSE each need additional columns/behave differently when this is set. |
| `mode` | `download` | — | `download` performs the actual pixel download; `info` only computes/records metadata (cloud %, snow/ice %, product IDs) without downloading. **GEE backend only** — no effect on GCLD/STAC/CDSE. |
| `target` | `all` | — | `water` restricts cloud/snow-ice percentage statistics to water pixels only (via a JRC Global Surface Water mask). **GEE backend only.** |
| `cloud_percentage` | `100` | — | Cloud-cover threshold (%); acquisitions above it are skipped. **GEE** (and partially **STAC**, which reads the same key) — no effect on GCLD/CDSE. |
| `snowice_percentage` | `100` | — | Snow/ice threshold (%); acquisitions above it are skipped. **GEE backend only.** |
| `output_mode` | `local` | — | `local` or `cloud`. Not read by the base class or by GEE/GCLD at all. Only the vendored STAC downloader defines this option, and even there it's **not actually wired up** from `download.yaml` — `STACDownloader` doesn't forward `output_mode` into the STAC engine's config, so it always runs in `local` mode regardless of what you set here. |
| `output_format` | `cog` | — | `cog` or `geotiff`. Read by the STAC backend (forwarded as-is). **Not honored by GCLD** — GCLD/ACOLITE output is always COG regardless of this setting. Not read by GEE or CDSE. |
| `proj_data` | none | optional | Directory containing `proj.db`. Read directly out of the raw config file by `main.py` *before* any geo library is imported (bypasses the `Downloader` class entirely), to override `PROJ_DATA`/`PROJ_LIB` and avoid conflicts with other GDAL/PROJ installs (e.g. SeaDAS). |

### `GEE` section

Only meaningful when `backend: gee`. Keys are merged into `GLOBAL` as described above.

| Key | Default | Purpose |
|---|---|---|
| `project_id` | — (required) | Google Cloud / Earth Engine project id, passed to `ee.Initialize(project=...)`. If initialization fails, `ee.Authenticate()` runs automatically on first use. |
| `target` | see GLOBAL | Same key as `GLOBAL.target`, documented above; conventionally set here rather than in `GLOBAL`. |
| `mode` | see GLOBAL | Same key as `GLOBAL.mode`. |
| `cloud_percentage` | see GLOBAL | Same key as `GLOBAL.cloud_percentage`. |
| `snowice_percentage` | see GLOBAL | Same key as `GLOBAL.snowice_percentage`. |
| `min_aoi_coverage` | `0` | Percent (0-100) of the AOI that must carry valid pixels for a merged GeoTIFF to actually be written. `0` keeps every acquisition, even ones that only clip the AOI's edge; raising it drops low-coverage overpasses (they're skipped, not written as partial files). |

Note: the S2/Landsat cloud & snow-ice query always samples at a hardcoded 20 m resolution, independent of each asset's own `resolution` key.

### `GCLD` section

Downloads Sentinel-2 L1C SAFE products from the public GCS bucket and generates TOA reflectance via ACOLITE. Only meaningful when `backend: gcld`.

| Key | Default | Required | Purpose |
|---|---|---|---|
| `date_csv` | — | **yes, in practice** | The GCLD run loop groups work by `self.date_df['sensor']` with no fallback, so it will raise if `date_csv` isn't set — treat it as mandatory for this backend, even though the base class alone doesn't enforce it. Must contain **`name`, `date` (YYYYMMDD), `product_ids` (comma-separated), and `sensor`** columns. The in-file comment in `download.yaml` currently omits `sensor` — it's required. `sensor` values must match an asset-key prefix (`s2`, `lc08`, `lc09`, `s3`, `s1`, `alphaearth`). |
| `remove_downloaded` | `false` | — | If `true`, deletes the temp download directory (raw SAFE data) after all AOI cells for a given date have been processed, keeping only the generated TOA outputs. |
| `acolite_dir` | — | **yes** | Path to a local ACOLITE clone; prepended to `sys.path` so `import acolite` succeeds. All other ACOLITE settings (target resolution, AOI bounds, temp cleanup) are hardcoded by `GCLDDownloader`, not configurable via `download.yaml`. |

SAFE product paths are resolved directly from the product id (regex-extracted MGRS tile → `gs://gcp-public-data-sentinel-2/tiles/...` glob, resolved via `gsutil ls -d`) — no BigQuery config is needed for the normal path. A BigQuery fallback (`bigquery-public-data.cloud_storage_geo_index.sentinel_2_index`) only triggers if that direct lookup fails, using Application Default Credentials with no project/dataset key read from `download.yaml`.

### `STAC` section

Only meaningful when `backend: stac`. This backend wraps a vendored downloader with a **hardcoded** STAC endpoint list and a hardcoded per-asset collection map, so several of the keys below currently have no effect — see [Known limitations](#known-limitations--inert-config) before relying on them.

| Key | Default | Live? | Purpose |
|---|---|---|---|
| `stac_api` | — | **no** | Intended to pick the STAC endpoint. The vendored downloader ignores this and always tries a fixed list (Element84 Earth Search, then Microsoft Planetary Computer) in order. |
| `stac_best_item` | — | **no** | Not read anywhere in the current code. |
| `stac_max_items` | — | **no** | Not read anywhere in the current code. |
| `clip_aoi` | — | **no** | Typo/name mismatch — the code reads `clip_to_aoi`, not `clip_aoi`. As written in `download.yaml` today, this key is silently ignored and clipping defaults to off. |
| `merge_outputs` | `false` | yes | Whether per-tile/per-scene outputs are merged into one file. |
| `output_format` | `cog` | yes | Forwarded to the STAC engine as-is (`cog` or `zarr`-family formats it supports). |

Per-asset collection IDs (e.g. `S2_L1TOA` → `sentinel-2-l1c`) are similarly hardcoded in the vendored downloader rather than read from each asset's `stac_source` key — only `S2_L1TOA`, `S2_L2RGB`, `S2_L2SURF`, `S1_L1C`, `LC08_L1TOA`, `LC08_L2RGB` are supported by this backend today.

### `CDSE` section

Downloads Sentinel-3 OLCI L1 products from the Copernicus Data Space Ecosystem and generates TOA reflectance via ACOLITE, with optional cloud masking. Only meaningful when `backend: cdse`.

| Key | Default | Required | Purpose |
|---|---|---|---|
| `cdse_username` | — | **yes** | CDSE account email, used for OAuth2 password-grant authentication (`client_id=cdse-public`, a fixed public client — no client secret needed). |
| `cdse_password` | — | **yes** | CDSE account password. |
| `date_csv` | none | **optional** | Unlike GCLD, CDSE does **not** require this. If set, must contain `name`, `date` (YYYYMMDD), `product_ids` (CDSE UUIDs, comma-separated) — the run uses exactly those products. If omitted, CDSE auto-discovers products by AOI + date range via the OData catalog search instead. |
| `remove_downloaded` | `false` | — | If `true`, deletes extracted product folders (`.SEN3`) after TOA generation for that date. |
| `acolite_dir` | — | **yes** | Same mechanism as GCLD — path to a local ACOLITE clone, added to `sys.path`. |
| `cloud_masking` | `false` (string-compared) | — | If `true`, appends a 1-band classification band to each TOA GeoTIFF: `0`=clear land, `1`=clear water, `2`=cloud land, `3`=cloud water, `255`=invalid. |
| `cloud_buffer_size` | `2` | — | IdePix cloud buffer radius, in pixels (~300 m each at OLCI's native resolution). Ignored when `cloud_mask_method: native`. |
| `cloud_mask_method` | `idepix` | — | `idepix` tries ESA SNAP IdePix via `esa_snappy`, automatically falling back to `native` if the CPU lacks AVX2, `esa_snappy` isn't installed, or the IdePix OLCI operator isn't registered. `native` always uses the product's own `qualityFlags.nc` bits — no SNAP required. |

Note: `cloud_mask_sen3.py` at the repo root is a standalone diagnostic CLI script (positional args) — it is unrelated to these `CDSE:` config keys and isn't part of the `download.yaml`-driven pipeline.

### `ASSETS`

A list of asset definitions, each a single-key mapping: `- ASSET_NAME: {...}`. `ASSET_NAME`'s prefix before the first `_` (e.g. `s2`, `lc08`, `lc09`, `s3`, `s1`, `alphaearth`) determines its sensor type (`optical`, `radar`, `embedding`), which in turn selects the download logic.

Not every key is meaningful for every backend — this matrix shows what's actually read where:

| Key | GEE | GCLD | STAC | CDSE | Notes |
|---|:-:|:-:|:-:|:-:|---|
| `gee_source` | ✅ | — | — | — | GEE collection id, e.g. `COPERNICUS/S2_HARMONIZED`. |
| `stac_source` | — | — | ❌ | — | Present in the schema but ignored — the STAC backend uses its own hardcoded collection map instead. |
| `gcld_source` | — | ❌ | — | — | Leftover from before GCLD switched to product-id-based resolution; not read. |
| `cdse_source` | — | — | — | ❌ | Collection name is hardcoded (`SENTINEL-3`) in the CDSE backend; this key is not read. |
| `cdse_product_type` | — | — | — | ✅ | Selects the S3 product type (`ol_1_efr` default, or `ol_1_err`/`sl_1_rbt`/`sy_2_syn`). |
| `include_bands` | ✅ | — | ✅ | — | Comma-separated band list. Not read by GCLD or CDSE — CDSE's OLCI band set is hardcoded from ACOLITE's output. |
| `resolution` | ✅ | — | ✅ | — | Output pixel size in meters. Not read by GCLD or CDSE — CDSE always writes at OLCI's native 300 m. |
| `save_dir` | ✅ | ✅ | ✅ | ✅ | Subdirectory name under `<project save_dir>/`, e.g. `L1`, `L2`, `L2RGB`, `EMBEDDING`. |
| `anonym` | ✅ | ✅ | ✅ | ✅ | Short label used in the output path/filename, e.g. `s2_msi`, `l8_oli`, `s1_sar`. |
| `obs_geo_pixel` | ✅ | — | — | — | See [Known limitations](#known-limitations--inert-config) for a YAML-quoting gotcha. |
| `vmin` / `vmax` | ✅ (RGB assets only) | — | — | — | Rescale range for building a `uint8` false-color/true-color composite; required when the asset name contains `rgb`. |

---

## Output structure

```
<save_dir>/<aoi_file_stem>/<asset_savedir>/<anonym>/<aoi_cell_name>/<year>/
    <ASSET>_<acquisition_time>_<aoi_cell_name>_<resolution>m.tif
```

- `<save_dir>` is the raw `GLOBAL.save_dir` value.
- `<aoi_file_stem>` is the AOI shapefile's filename without extension (e.g. `wemendji` for `wemendji.shp`) — this segment is added automatically, not itself configurable.
- `<asset_savedir>` and `<anonym>` come from that asset's `save_dir`/`anonym` keys under `ASSETS:`.
- `<aoi_cell_name>` is the per-feature `name` from the AOI shapefile (or a generated index if no `name` column exists).
- `<resolution>` is that asset's configured (or, for CDSE, hardcoded) resolution.

S3 OLCI output GeoTIFFs contain 21 TOA reflectance bands (`rhot_*`), observing geometry angles (SZA/VZA/RAA), and — when `cloud_masking: true` — a classification band (`0`=clear land, `1`=clear water, `2`=cloud land, `3`=cloud water, `255`=invalid).

---

## Known limitations / inert config

A few `download.yaml` keys exist but currently have no effect, or have a non-obvious gotcha, due to bugs or leftover config from earlier refactors. Listed here so you don't spend time debugging a silent no-op:

- **`ASSETS.<asset>.obs_geo_pixel: true`** (unquoted YAML boolean) does **not** enable per-pixel observation geometry for GEE downloads. The code compares the value against the strings `'True'`/`'true'`, but PyYAML parses an unquoted `true` as a Python `bool`, so the comparison always fails. **Write it as a quoted string instead: `obs_geo_pixel: 'true'`.**
- **GCLD's `date_csv`** needs a `sensor` column in addition to `name`/`date`/`product_ids` — the in-file comment doesn't mention it, but the code requires it.
- **GCLD only supports Sentinel-2** (`sensor: s2`) in practice — the lookup functions for other sensors (`lc08`, `lc09`, `s3`, `s1`) aren't implemented for this backend and would raise an error if referenced from a `date_csv` row.
- **GCLD ignores `GLOBAL.output_format`** — TOA output is always a COG.
- **GCLD's `temp_download_dir`** (if you tried setting it) has no effect — it's looked up at the wrong level of the internal config dict and always falls back to `<save_dir>/temp_dir/<date>`.
- **STAC's `clip_aoi`** doesn't do anything — the code expects `clip_to_aoi`. **STAC's `stac_api`, `stac_best_item`, `stac_max_items`** are also not read; the backend always tries a fixed endpoint list (Element84 Earth Search, then Planetary Computer) and looks up each asset's collection from a hardcoded map.
- **`ASSETS.S3_L1TOA.include_bands`, `.resolution`, and `.cdse_source`** have no effect under the CDSE backend — OLCI's band set and 300 m resolution are hardcoded, and the queried collection is always `SENTINEL-3`.
- **CDSE's `date_csv` is optional**, unlike GCLD's — if you don't set it, CDSE auto-discovers products via the OData catalog by AOI + date range instead of requiring a pre-built list.
