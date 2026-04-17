# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geospatial food access analysis pipeline for San Francisco. It computes walking-time accessibility to grocery stores for every Census block, accounting for elevation-adjusted walking speeds via the Tobler hiking function.

## Setup & Running

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (bash)
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Run full pipeline (~5–15 min first run due to DEM download)
python tobler_food_access.py

# Validate outputs
python validate_tobler.py
```

There is no test suite. There are no build steps.

## Pipeline Architecture

`tobler_food_access.py` runs a sequential 10-step pipeline:

1. **OSM Network** — Download San Francisco walk network via OSMnx
2. **DEM Cache** — Fetch/cache USGS 10m elevation raster (`sf_dem_10m.tif`) via py3dep
3. **Tobler Edge Costs** — Assign per-directed-edge travel times using slope-adjusted walking speed: `v = 6 × exp(−3.5 × |grade + 0.05|)` km/h
4. **USDA Retailers** — Download grocery store locations from ArcGIS Hub API
5. **Census Blocks** — Fetch block geometries + population from TIGERweb REST API
6. **Node Snapping** — Snap store and block centroid points to nearest OSM graph nodes
7. **Dual-Sweep Dijkstra** — ~100 routing calls (2 per store) using igraph C backend; outbound (unloaded) + return (1.2× grocery penalty); much faster than per-block routing
8. **Tract Rollup** — Aggregate block-level accessibility scores to Census tract level
9. **Static Map** — 2-panel choropleth + histogram via Matplotlib → `food_access_map.png`
10. **Interactive Map** — Block-level Folium HTML map → `food_access_blocks.html`

**Key output:** `food_access_by_tract.geojson` — tract polygons with accessibility metrics.

## Key Constants (hardcoded in `tobler_food_access.py`)

| Constant | Value | Meaning |
|---|---|---|
| `PLACE` | `"San Francisco, California, USA"` | OSMnx query target |
| `THRESHOLD_SEC` | `1800` | 30-min round-trip walk cutoff |
| `GROCERY_PENALTY` | `1.2` | Return-trip slowdown factor |
| `DEM_PATH` | `sf_dem_10m.tif` | Local DEM cache filename |
| `STORE_TYPES` | Grocery/Super/Supermarket | USDA retailer filter |

## External Data Sources

- **OpenStreetMap** via OSMnx (walk network)
- **USGS 3DEP** via py3dep (10m elevation DEM)
- **USDA ArcGIS Hub** REST API (retailer locations)
- **Census TIGERweb** REST API (block/tract geometry + population)

All data is fetched at runtime; the DEM is cached locally after first download. `outputs/` and `*.tif` files are gitignored.

## igraph vs NetworkX

igraph (C backend) is the primary routing engine for ~10–100× speedup. NetworkX is a fallback if igraph is unavailable. The dual-sweep approach avoids the O(blocks × stores) routing explosion.
