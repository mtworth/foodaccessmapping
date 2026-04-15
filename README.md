# Food Access Mapping — SF

Slope-adjusted, round-trip walking time to groceries per Census block, rolled up to Census tract-level percent food access for San Francisco, CA.

## What It Does

The USDA food access methodology has two significant flaws for a city like San Francisco: it measures straight-line distance from a tract centroid rather than actual walking routes, and it ignores topography entirely. San Francisco's extreme terrain means a store 0.5 miles away can require 25 minutes to reach on foot if there's a 200-foot climb involved.

This project addresses both problems:

1. **Real network routing** — uses the OSM pedestrian network (via OSMnx) to compute actual walk distances along real streets, not crow-flies buffers.
2. **Hill-aware speed via Tobler's hiking function** — adjusts walking speed per directed edge based on measured slope from a USGS 10m DEM. Steeper grades reduce speed; descents are faster than flat ground but not free.

The output is the percentage of each Census tract's population with a qualifying grocery store within a 30-minute round-trip walk, accounting for both route distance and elevation change.

## Methodology

### Step 1 — OSM Pedestrian Network

OSMnx downloads the walkable street network for San Francisco from OpenStreetMap as a directed `MultiDiGraph`. The graph is projected to UTM Zone 10N (EPSG:32610) for metric edge lengths. Each street produces two directed edges (one per direction), which is essential for correct slope handling.

### Step 2 — USGS 10m DEM

`py3dep` fetches the USGS 3D Elevation Program (3DEP) DEM at 10-meter resolution for the graph bounding box. The raster is saved locally (`sf_dem_10m.tif`) and reused on subsequent runs.

### Step 3 — Tobler Edge Costs

Node elevations are sampled from the DEM via `ox.elevation.add_node_elevations_raster`. Per-edge grade is computed as `(elev_to - elev_from) / length`. Tobler's hiking function is then applied to every directed edge:

```
v = (6 × exp(−3.5 × |grade + 0.05|)) / 3.6   [km/h → m/s]
```

Two time weights are stored on each edge:

- `time_out` — unloaded Tobler time (block → store, outbound)
- `time_back` — `time_out × 1.2` (store → block, return with groceries)

The 1.2x grocery penalty approximates the speed reduction from carrying a loaded bag. Slope asymmetry on the return trip requires no special handling: the `v→u` directed edge already carries the reversed grade relative to `u→v`, so routing store→block on the directed graph automatically applies the correct uphill/downhill Tobler speed in each direction.

**Missing elevation assumption:** Some OSM nodes near water bodies, parks, or at the edge of the raster extent do not receive a valid elevation value from the DEM. These nodes produce `grade = NaN` for incident edges, which igraph's Dijkstra rejects. The fallback treats any NaN grade as 0.0 (flat ground) and any NaN edge length as 1.0 m. This is conservative — it slightly underestimates access barriers only in the rare cases where a steep edge falls exactly outside the raster boundary.

### Step 4 — USDA Retailers

EBT-approved grocery retailers (Grocery Store, Super Store, Supermarket) for San Francisco County are downloaded from the USDA ArcGIS Hub.

### Step 5 — Census Blocks

Census block geometries and decennial population counts (`POP100`) for San Francisco County (FIPS: 06075) are fetched from the Census TIGERweb REST API.

### Step 6 — Routing

Block centroids and store locations are snapped to their nearest graph nodes. For each store, two single-source Dijkstra sweeps cover the full graph:

- **Outbound sweep** — traverses the directed graph in reverse (`mode=IN`) with weight `time_out`, yielding the least-cost path from every block centroid to that store.
- **Return sweep** — traverses forward (`mode=OUT`) with weight `time_back`, yielding the least-cost path from that store back to every block centroid.

Round-trip time is the sum of the two sweeps. Each block records the minimum round-trip time across all stores. This is ~2N Dijkstra calls (where N is the number of stores) rather than one call per block-store pair.

igraph (C backend) is used when available for ~10-100x speedup over NetworkX. NetworkX is the fallback.

### Step 7 — Accessibility Threshold

A block is **accessible** if its minimum round-trip time is ≤ 30 minutes (1,800 seconds).

### Step 8 — Population-Weighted Tract Rollup

Block-level results are aggregated to Census tracts:

```
pct_accessible = (sum of POP100 for accessible blocks in tract) /
                 (sum of POP100 for all blocks in tract) × 100
```

Tracts with zero population are marked `NaN`.

## Why This Improves on USDA

| Approach | Distance measure | Terrain |
|---|---|---|
| USDA (baseline) | Straight-line from tract centroid | Ignored |
| This project (v1 prototype) | 15-min walk isochrones via OpenRouteService | Ignored |
| This project (current) | Dijkstra on OSM pedestrian graph | Tobler per directed edge |

For San Francisco specifically:

- **Network routing matters** because the street grid is irregular and many blocks are bounded by stairways or one-way pedestrian paths that a radius buffer cannot represent.
- **Elevation matters** because the city's hills routinely add 50–100% to the effective walking time versus flat-terrain assumptions. The Tenderloin is effectively closer to a Nob Hill store in straight-line terms than in walk-time terms. Tobler quantifies that difference per edge.
- **Directed graph matters** because a block east of a steep ridge has a fast downhill outbound trip and a slow uphill return. Treating the slope as symmetric would undercount round-trip cost in exactly the neighborhoods where access is already constrained.

## Files

| File | Description |
|---|---|
| `tobler_food_access.py` | Main script — full pipeline from data download to output files |
| `validate_tobler.py` | Validates the Tobler function with San Francisco geography examples |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated files (gitignored); created automatically on first run |

## Setup & Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tobler_food_access.py
```

The DEM download (~300MB) and routing step are the two slow stages. The DEM is cached to `sf_dem_10m.tif` after the first run. Routing runtime scales with the number of stores (~100 for SF) and is on the order of minutes with igraph.

## Output

| File | Contents |
|---|---|
| `outputs/food_access_by_tract.geojson` | Census tract polygons with `total_pop`, `accessible_pop`, and `pct_accessible` columns |
| `outputs/food_access_map.png` | Two-panel figure: choropleth of tract-level access (5 discrete bins, RdYlGn) and histogram of tract distribution with city-wide mean |
| `outputs/food_access_blocks.html` | Interactive Folium map at Census-block resolution. Block polygons colored by continuous round-trip walk time (green → red, 0–30 min); grey = unreachable. Toggle to store-count layer (Blues scale). Hover any block to see its GEOID, population, travel time, and stores in range. Open in any browser. |

## Dependencies

```
osmnx
networkx
igraph
geopandas
pandas
numpy
rasterio
requests
shapely
py3dep
rioxarray
matplotlib
rasterstats
folium
branca
```

## Data Sources

| Source | Purpose |
|---|---|
| [OpenStreetMap via OSMnx](https://osmnx.readthedocs.io/) | Pedestrian network |
| [USGS 3DEP via py3dep](https://github.com/hyriver/py3dep) | 10m elevation raster |
| [USDA ArcGIS Hub](https://opendata.arcgis.com/) | EBT-approved retailer locations |
| [Census TIGERweb](https://tigerweb.geo.census.gov/) | Block geometries and population counts |
