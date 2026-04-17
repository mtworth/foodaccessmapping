"""
tobler_food_access.py

Slope-adjusted, round-trip walking time to groceries for each SF Census block.
Results aggregated to Census tracts as % of population with food access.

Methodology
-----------
- Pedestrian network: OpenStreetMap via OSMnx (directed MultiDiGraph)
- Elevation: USGS 10m 3DEP DEM via py3dep
- Speed model: Tobler's hiking function applied per directed edge
    v = (6 * exp(-3.5 * |grade + 0.05|)) / 3.6  [km/h → m/s]
- Outbound:  Dijkstra(block_centroid → store, weight='time_out')
- Return:    Dijkstra(store → block_centroid, weight='time_back')
    time_back = time_out × 1.2  (grocery load penalty)
    Slope asymmetry on the return trip is handled implicitly — each
    directed edge u→v has grade (elev_v - elev_u)/length; its paired
    reverse edge v→u carries the opposite grade. Routing store→block
    on the directed graph naturally traverses uphill/downhill edges
    with the correct Tobler speed in each direction.
- Accessible = round-trip ≤ 30 minutes
- Tract metric = accessible_population / total_population × 100

Dependencies
------------
    pip install osmnx networkx igraph geopandas pandas numpy rasterio
                requests shapely py3dep rioxarray matplotlib folium branca
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLACE = "San Francisco, California, USA"
STATE_FIPS = "06"
COUNTY_FIPS = "075"

USDA_DATASET_ID = "8b260f9a10b0459aa441ad8588c2251c_0"
STORE_TYPES = ["Grocery Store", "Super Store", "Supermarket"]

DEM_PATH = "sf_dem_10m.tif"
THRESHOLD_SEC = 30 * 60   # 30-minute round-trip threshold
GROCERY_PENALTY = 1.2     # return-trip speed reduction (carrying groceries)

OUTPUT_DIR = "outputs"
OUTPUT_GEOJSON = f"{OUTPUT_DIR}/food_access_by_tract.geojson"
OUTPUT_MAP = f"{OUTPUT_DIR}/food_access_map.png"
OUTPUT_BLOCK_MAP = f"{OUTPUT_DIR}/food_access_blocks.html"
OUTPUT_TRACT_MAP    = f"{OUTPUT_DIR}/food_access_tracts.html"
OUTPUT_PRIORITY_MAP = f"{OUTPUT_DIR}/priority_map.html"


# ---------------------------------------------------------------------------
# Step 1 — Build walk graph
# ---------------------------------------------------------------------------

def build_walk_graph(place):
    """Download SF pedestrian network from OSM and project to UTM metres."""
    print(f"Downloading OSM walk network for {place} ...")
    G = ox.graph_from_place(place, network_type="walk")
    G = ox.project_graph(G)  # EPSG:32610 (UTM Zone 10N) for SF
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"  {len(nodes):,} nodes  |  {len(edges):,} edges")
    return G


# ---------------------------------------------------------------------------
# Step 2 — Download USGS 10m DEM
# ---------------------------------------------------------------------------

def download_dem(G, dem_path):
    """Fetch USGS 10m 3DEP DEM for the graph bounding box via py3dep."""
    try:
        import py3dep
        import rioxarray  # noqa: F401 — required by py3dep for .rio accessor
    except ImportError:
        raise ImportError(
            "py3dep and rioxarray are required for DEM download.\n"
            "  pip install py3dep rioxarray"
        )

    nodes = ox.graph_to_gdfs(G, edges=False).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = nodes.total_bounds
    bbox = box(minx, miny, maxx, maxy)

    print("Downloading USGS 10m DEM (may take a minute) ...")
    dem = py3dep.get_dem(bbox, resolution=10)
    dem.rio.to_raster(dem_path)
    print(f"  DEM saved to {dem_path}")


# ---------------------------------------------------------------------------
# Step 3 — Elevation + Tobler edge costs
# ---------------------------------------------------------------------------

def add_tobler_costs(G, dem_path, grocery_penalty=GROCERY_PENALTY):
    """
    Attach slope-aware travel times to every directed edge.

    time_out  — unloaded Tobler time (outbound, block → store)
    time_back — loaded Tobler time × grocery_penalty (return, store → block)

    Because OSMnx creates edges in both directions for walkable streets,
    routing store→block with time_back automatically uses the reversed-slope
    speed for each edge — no explicit slope-flip needed.
    """
    print("Sampling node elevations from DEM ...")
    G = ox.elevation.add_node_elevations_raster(G, dem_path)
    # Adds 'grade' = (elev_to - elev_from) / length per directed edge
    G = ox.elevation.add_edge_grades(G, add_absolute=True)

    def tobler_ms(grade):
        """Tobler hiking function → speed in m/s."""
        return (6.0 * np.exp(-3.5 * abs(grade + 0.05))) / 3.6

    for _, _, _, data in G.edges(data=True, keys=True):
        length = float(np.nan_to_num(data.get("length", 1.0), nan=1.0))
        raw_grade = data.get("grade", 0.0)
        grade = float(np.nan_to_num(raw_grade, nan=0.0))
        data["time_out"] = length / tobler_ms(grade)
        data["time_back"] = data["time_out"] * grocery_penalty

    print("  Tobler edge costs assigned.")
    return G


# ---------------------------------------------------------------------------
# Step 4 — Load USDA retailers
# ---------------------------------------------------------------------------

def load_retailers():
    """Download EBT-approved grocery retailers for SF from USDA ArcGIS Hub."""
    url = (
        f"https://opendata.arcgis.com/api/v3/datasets/{USDA_DATASET_ID}"
        f"/downloads/data?format=geojson&spatialRefId=4326"
    )
    gdf = gpd.read_file(url)
    subset = gdf[
        (gdf["State"] == "CA")
        & (gdf["County"] == "SAN FRANCISCO")
        & (gdf["Store_Type"].isin(STORE_TYPES))
    ].copy()
    print(f"Loaded {len(subset)} qualifying retailers")
    return subset


# ---------------------------------------------------------------------------
# Step 5 — Load Census blocks with population counts
# ---------------------------------------------------------------------------

def load_census_blocks():
    """
    Fetch Census blocks for SF from TIGERweb.
    Retains POP100 (decennial count) for population-weighted tract rollup.
    """
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/"
        "TIGERweb/Tracts_Blocks/MapServer/2/query"
    )
    params = {
        "where": f"STATE='{STATE_FIPS}' AND COUNTY='{COUNTY_FIPS}'",
        "outFields": "*",
        "f": "geojson",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    gdf = gpd.GeoDataFrame.from_features(r.json()["features"])
    gdf = gdf.set_crs("EPSG:4326")
    gdf["BLOCK_GEOID"] = gdf["GEOID"]
    gdf["TRACT_GEOID"] = gdf["GEOID"].str[:11]
    gdf["POP100"] = pd.to_numeric(gdf["POP100"], errors="coerce").fillna(0).astype(int)

    print(
        f"Loaded {len(gdf):,} Census blocks  |  "
        f"{gdf['POP100'].sum():,} total population"
    )
    return gdf[["BLOCK_GEOID", "TRACT_GEOID", "POP100", "geometry"]].copy()


# ---------------------------------------------------------------------------
# Step 5b — Load tract-level poverty estimates (ACS 5-year)
# ---------------------------------------------------------------------------

def load_tract_poverty():
    """
    Fetch income-to-poverty-ratio estimates for SF Census tracts from ACS 5-year.

    Uses table C17002 (Ratio of Income to Poverty Level):
      C17002_001E — total population for whom ratio is determined
      C17002_002E — under 0.50  (< 50% FPL)
      C17002_003E — 0.50–0.99   (50–99% FPL)
      C17002_004E — 1.00–1.24
      C17002_005E — 1.25–1.49
      C17002_006E — 1.50–1.74
      C17002_007E — 1.75–1.99
      C17002_008E — 2.00 and over

    pct_poverty = population in bands _002–_007 (< 200% FPL) / total × 100

    Returns a DataFrame with columns:
        TRACT_GEOID  — 11-digit GEOID
        pct_poverty  — % of tract population below 200% of the federal poverty line
    """
    POVERTY_CACHE = os.path.join(OUTPUT_DIR, "_poverty_cache.csv")
    if os.path.exists(POVERTY_CACHE):
        df = pd.read_csv(POVERTY_CACHE, dtype={"TRACT_GEOID": str})
        print(f"Loaded 200% FPL estimates from cache ({len(df):,} SF tracts  |  "
              f"city-wide median: {df['pct_poverty'].median():.1f}%)")
        return df[["TRACT_GEOID", "pct_poverty"]].copy()

    fields = ",".join([f"C17002_{str(i).zfill(3)}E" for i in range(1, 9)])
    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": fields,
        "for": "tract:*",
        "in": f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    df["TRACT_GEOID"] = df["state"] + df["county"] + df["tract"]
    for col in [f"C17002_{str(i).zfill(3)}E" for i in range(1, 9)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_poverty_universe"] = df["C17002_001E"]
    # Sum bands 002–007: everyone below 200% FPL
    below_200pct = [f"C17002_{str(i).zfill(3)}E" for i in range(2, 8)]
    df["pop_below_200pct_fpl"] = df[below_200pct].sum(axis=1)

    df["pct_poverty"] = np.where(
        df["total_poverty_universe"] > 0,
        df["pop_below_200pct_fpl"] / df["total_poverty_universe"] * 100,
        np.nan,
    )

    result = df[["TRACT_GEOID", "pct_poverty"]].copy()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.to_csv(POVERTY_CACHE, index=False)
    print(f"Loaded 200% FPL estimates for {len(result):,} SF tracts  |  "
          f"city-wide median: {result['pct_poverty'].median():.1f}%")
    return result


# ---------------------------------------------------------------------------
# Step 6 — Snap points to nearest graph nodes
# ---------------------------------------------------------------------------

def snap_to_graph(G, gdf_points, graph_crs):
    """Return a list of nearest graph node IDs for each point geometry."""
    pts = gdf_points.to_crs(graph_crs)
    return ox.nearest_nodes(G, pts.geometry.x.tolist(), pts.geometry.y.tolist())


# ---------------------------------------------------------------------------
# Step 7 — Compute minimum round-trip time per block
# ---------------------------------------------------------------------------

def compute_min_roundtrip(G, block_nodes, store_nodes):
    """
    Store-centric single-source Dijkstra: for each store run two sweeps across
    the entire graph, then look up every block's cost in O(1).

    ~100 Dijkstra calls total (2 per store) vs ~60,000 in the block-centric
    approach. Uses igraph (C backend) when available, falls back to NetworkX.

    Outbound sweep  — mode IN  on directed graph, weight='time_out'
                      gives cost of every block → this store path.
    Return sweep    — mode OUT on directed graph, weight='time_back'
                      gives cost of this store → every block path.
                      Slope asymmetry is implicit: the v→u directed edge
                      already carries the reversed grade.

    Returns a list aligned with block_nodes; None where no path exists.
    """
    unique_stores = list(dict.fromkeys(store_nodes))
    n_stores = len(unique_stores)

    # Both implementations return (min_rt_list, store_count_list)
    try:
        import igraph as ig  # noqa: F401
        return _compute_igraph(G, block_nodes, unique_stores)
    except ImportError:
        print("  igraph not found - falling back to NetworkX (slower).")
        print("  Install with: pip install igraph")
        return _compute_networkx(G, block_nodes, unique_stores)


def _build_igraph(G):
    """Manually convert OSMnx MultiDiGraph to igraph (osmnx 2.x compatible)."""
    import igraph as ig

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    edges, weights_out, weights_back = [], [], []
    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights_out.append(data.get("time_out", 1.0))
        weights_back.append(data.get("time_back", 1.0))

    G_ig = ig.Graph(n=len(node_list), edges=edges, directed=True)
    G_ig.vs["osmid"] = node_list
    G_ig.es["time_out"] = weights_out
    G_ig.es["time_back"] = weights_back
    return G_ig, node_to_idx


def _compute_igraph(G, block_nodes, unique_stores):
    """igraph implementation — C backend, ~10–100x faster than NetworkX."""
    G_ig, osmid_to_idx = _build_igraph(G)

    store_idxs = [osmid_to_idx[n] for n in unique_stores]
    block_idxs = [osmid_to_idx[n] for n in block_nodes]

    INF = float("inf")
    min_rt = [INF] * len(block_nodes)
    store_count = [0] * len(block_nodes)

    for i, (store_node, s_idx) in enumerate(zip(unique_stores, store_idxs)):
        print(f"  Store {i + 1} / {len(unique_stores)}  (node {store_node}) ...")

        # Outbound: cost from each block TO this store
        # mode='IN' traverses incoming edges → equivalent to reverse Dijkstra
        out_dists = G_ig.distances(s_idx, weights="time_out", mode="IN")[0]

        # Return: cost from this store TO each block (with grocery penalty)
        back_dists = G_ig.distances(s_idx, weights="time_back", mode="OUT")[0]

        for j, b_idx in enumerate(block_idxs):
            t_out = out_dists[b_idx]
            t_back = back_dists[b_idx]
            if t_out < INF and t_back < INF:
                rt = t_out + t_back
                if rt < min_rt[j]:
                    min_rt[j] = rt
                if rt <= THRESHOLD_SEC:
                    store_count[j] += 1

    return [t if t < INF else None for t in min_rt], store_count


def _compute_networkx(G, block_nodes, unique_stores):
    """NetworkX fallback using single-source Dijkstra from each store."""
    G_rev = G.reverse(copy=False)  # reversed graph for inbound sweep
    INF = float("inf")
    min_rt = [INF] * len(block_nodes)
    store_count = [0] * len(block_nodes)

    for i, store_node in enumerate(unique_stores):
        print(f"  Store {i + 1} / {len(unique_stores)}  (node {store_node}) ...")

        # Outbound: reverse graph, source=store → gives cost of block→store paths
        out_dists = nx.single_source_dijkstra_path_length(
            G_rev, store_node, weight="time_out"
        )
        # Return: forward graph, source=store → gives cost of store→block paths
        back_dists = nx.single_source_dijkstra_path_length(
            G, store_node, weight="time_back"
        )

        for j, block_node in enumerate(block_nodes):
            t_out = out_dists.get(block_node, INF)
            t_back = back_dists.get(block_node, INF)
            if t_out < INF and t_back < INF:
                rt = t_out + t_back
                if rt < min_rt[j]:
                    min_rt[j] = rt
                if rt <= THRESHOLD_SEC:
                    store_count[j] += 1

    return [t if t < INF else None for t in min_rt], store_count


# ---------------------------------------------------------------------------
# Step 8 — Roll up to Census tracts
# ---------------------------------------------------------------------------

def load_cartographic_tracts():
    """
    Fetch SF tract boundaries pre-clipped to the coastline from the Census
    cartographic boundary file (cb_2020_06_tract_500k). These align with actual
    land — no ocean bleed, Treasure Island included correctly.

    Cached to outputs/_tract_cb.gpkg after first download.
    """
    cache = os.path.join(OUTPUT_DIR, "_tract_cb.gpkg")
    if os.path.exists(cache):
        gdf = gpd.read_file(cache)
        print(f"  Loaded cartographic tract boundaries from cache ({len(gdf)} tracts)")
        return gdf

    url = ("https://www2.census.gov/geo/tiger/GENZ2020/shp/"
           "cb_2020_06_tract_500k.zip")
    print("  Downloading Census cartographic tract boundaries ...")
    gdf = gpd.read_file(url)
    sf  = gdf[gdf["COUNTYFP"] == COUNTY_FIPS].copy()
    sf["TRACT_GEOID"] = sf["STATEFP"] + sf["COUNTYFP"] + sf["TRACTCE"]
    sf  = sf[["TRACT_GEOID", "geometry"]].to_crs("EPSG:4326").reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sf.to_file(cache, driver="GPKG")
    print(f"  Saved {len(sf)} SF cartographic tract boundaries to cache")
    return sf


def rollup_to_tracts(blocks_gdf):
    """
    Aggregate block-level accessibility to Census tracts.

    Uses Census cartographic boundary geometries (water-clipped) so tract
    polygons align with actual land — no bay/ocean bleed.

    Produces:
      total_pop           — total population in tract
      pct_accessible      — % pop with access to >=1 store (30-min round trip)
      pct_access_1store   — same as pct_accessible (alias for map layer naming)
      pct_access_2store   — % pop with access to >=2 stores
      pct_access_3store   — % pop with access to >=3 stores
      pct_access_4store   — % pop with access to >=4 stores
      pct_access_5store   — % pop with access to >=5 stores
    """
    # Stats from blocks (TIGER geometry — fine for population aggregation)
    stats = (
        blocks_gdf.groupby("TRACT_GEOID")["POP100"]
        .sum()
        .reset_index()
        .rename(columns={"POP100": "total_pop"})
    )

    # Swap in cartographic (coastline-clipped) geometries
    cb = load_cartographic_tracts()
    tracts = cb.merge(stats, on="TRACT_GEOID", how="left")
    tracts["total_pop"] = tracts["total_pop"].fillna(0).astype(int)

    # Per-threshold accessibility columns
    for n in range(1, 6):
        col = f"pct_access_{n}store"
        pop_n = (
            blocks_gdf[blocks_gdf["store_count"] >= n]
            .groupby("TRACT_GEOID")["POP100"]
            .sum()
        )
        tracts = tracts.merge(
            pop_n.rename(f"pop_{n}store"), on="TRACT_GEOID", how="left"
        )
        tracts[f"pop_{n}store"] = tracts[f"pop_{n}store"].fillna(0).astype(int)
        tracts[col] = np.where(
            tracts["total_pop"] > 0,
            tracts[f"pop_{n}store"] / tracts["total_pop"] * 100,
            np.nan,
        )

    # Keep pct_accessible as alias for ≥1 store (used by make_map static figure)
    tracts["pct_accessible"] = tracts["pct_access_1store"]

    city_pop = tracts["total_pop"].sum()
    city_acc = tracts["pop_1store"].sum()
    print(f"\nCity-wide: {city_acc:,} / {city_pop:,} people with >=1 store access "
          f"({city_acc / city_pop * 100:.1f}%)")
    for n in range(1, 6):
        pct = tracts[f"pct_access_{n}store"].mean()
        print(f"  Avg tract % with >={n} store(s): {pct:.1f}%")

    return tracts


# ---------------------------------------------------------------------------
# Step 9 — Map
# ---------------------------------------------------------------------------

def make_map(tract_gdf, retailer_gdf, output_path=OUTPUT_MAP, blocks_gdf=None):
    """
    Two-panel figure:
      Left  — choropleth of pct_accessible across Census tracts (5 discrete bins).
      Right — histogram of pct_accessible distribution with city-wide mean line.

    Parameters
    ----------
    tract_gdf     : GeoDataFrame with 'pct_accessible' column (0-100).
    retailer_gdf  : GeoDataFrame of grocery store point locations.
    output_path   : File path for the saved PNG.
    blocks_gdf    : Reserved for future block-level panel; currently unused.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    import matplotlib.ticker as mticker

    # --- Discrete 5-bin colour scheme (shared by both panels) ---
    bin_edges = [0, 20, 40, 60, 80, 100]
    bin_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    cmap_base = plt.get_cmap("RdYlGn", 5)
    bin_colors = [cmap_base(i) for i in range(5)]
    cmap5 = ListedColormap(bin_colors)
    norm5 = BoundaryNorm(bin_edges, ncolors=5)

    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2,
        figsize=(18, 11),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    fig.suptitle(
        "San Francisco Food Access by Census Tract\n"
        "OSM network  ·  USGS 10m DEM  ·  Tobler hiking function  ·  30-min round trip",
        fontsize=13,
        y=0.98,
    )

    # ------------------------------------------------------------------ #
    # Left panel — choropleth                                              #
    # ------------------------------------------------------------------ #
    tract_gdf.plot(
        column="pct_accessible",
        ax=ax_map,
        cmap=cmap5,
        norm=norm5,
        edgecolor="white",
        linewidth=0.3,
        missing_kwds={"color": "lightgrey", "label": "No population"},
        legend=False,
    )

    retailer_gdf.to_crs(tract_gdf.crs).plot(
        ax=ax_map,
        color="black",
        markersize=5,
        zorder=5,
        label="Grocery store",
    )

    # Build a tidy discrete legend for the choropleth
    legend_handles = [
        Patch(facecolor=bin_colors[i], edgecolor="grey", linewidth=0.4, label=bin_labels[i])
        for i in range(5)
    ]
    legend_handles.append(
        Patch(facecolor="lightgrey", edgecolor="grey", linewidth=0.4, label="No population")
    )
    store_handle = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="black",
        markersize=5, label="Grocery store",
    )
    legend_handles.append(store_handle)

    ax_map.legend(
        handles=legend_handles,
        title="% pop. within 30-min\nround-trip walk",
        title_fontsize=8,
        fontsize=8,
        loc="lower left",
        framealpha=0.88,
    )
    ax_map.set_title("% Population with Food Access (Census Tracts)", fontsize=11, pad=8)
    ax_map.set_axis_off()

    # ------------------------------------------------------------------ #
    # Right panel — histogram                                              #
    # ------------------------------------------------------------------ #
    valid = tract_gdf["pct_accessible"].dropna()
    city_mean = valid.mean()

    # Draw one bar per 20-point bin, coloured by the same RdYlGn palette
    bin_counts, _ = np.histogram(valid, bins=bin_edges)
    bin_centers = [10, 30, 50, 70, 90]
    bar_width = 18  # slightly narrower than 20 to show gaps

    for i, (count, color) in enumerate(zip(bin_counts, bin_colors)):
        ax_hist.bar(
            bin_centers[i],
            count,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=bin_labels[i],
        )

    # Vertical line at city-wide mean
    ax_hist.axvline(city_mean, color="steelblue", linewidth=2, linestyle="--", zorder=5)
    ax_hist.text(
        city_mean + 1.5,
        ax_hist.get_ylim()[1] if ax_hist.get_ylim()[1] > 0 else max(bin_counts) * 0.95,
        f"City mean\n{city_mean:.1f}%",
        color="steelblue",
        fontsize=9,
        va="top",
    )

    ax_hist.set_xlim(0, 100)
    ax_hist.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_hist.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_hist.set_xlabel("% Population with Food Access", fontsize=10)
    ax_hist.set_ylabel("Number of Census Tracts", fontsize=10)
    ax_hist.set_title("Distribution of Tract-Level Food Access", fontsize=11, pad=8)
    ax_hist.tick_params(axis="both", labelsize=9)
    ax_hist.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_hist.spines[["top", "right"]].set_visible(False)

    # Re-annotate the mean line now that y-limits are set after bars are drawn
    ymax = ax_hist.get_ylim()[1]
    # Clear existing text and redraw with correct y position
    for txt in ax_hist.texts:
        txt.remove()
    ax_hist.text(
        city_mean + 1.5,
        ymax * 0.95,
        f"City mean\n{city_mean:.1f}%",
        color="steelblue",
        fontsize=9,
        va="top",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Map saved to {output_path}")


# ---------------------------------------------------------------------------
# Step 10 — Interactive block-level map (Folium)
# ---------------------------------------------------------------------------

def make_block_map(blocks_gdf, retailer_gdf, output_path=OUTPUT_BLOCK_MAP):
    """
    Interactive Folium choropleth at Census-block resolution.

    Two toggleable layers:
      - Travel time: blocks colored green → red by round-trip minutes (0–30).
        Unreachable blocks are grey.
      - Store count: blocks colored by number of stores reachable within the
        30-min threshold (sequential Blues scale).

    Hover any block to see its GEOID, population, travel time, and store count.
    Grocery stores are plotted as circle markers visible on both layers.
    """
    import folium
    import branca.colormap as cm

    blocks_wgs = blocks_gdf.to_crs(4326).copy()
    stores_wgs = retailer_gdf.to_crs(4326)

    center = [
        blocks_wgs.geometry.centroid.y.mean(),
        blocks_wgs.geometry.centroid.x.mean(),
    ]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # ------------------------------------------------------------------ #
    # Layer 1 — Travel time (green → red)                                 #
    # ------------------------------------------------------------------ #
    vmax_time = 30.0
    time_colormap = cm.LinearColormap(
        colors=["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=0,
        vmax=vmax_time,
        caption="Round-trip walk time to nearest grocery store (minutes)",
    )

    def time_style(feature):
        val = feature["properties"].get("min_rt_min")
        color = "#aaaaaa" if val is None else time_colormap(min(float(val), vmax_time))
        return {"fillColor": color, "color": "#444444", "weight": 0.2, "fillOpacity": 0.75}

    tooltip_fields = ["BLOCK_GEOID", "POP100", "min_rt_min", "store_count", "accessible"]
    tooltip_aliases = ["Block GEOID", "Population", "Round-trip (min)", "Stores in range", "Accessible"]

    time_layer = folium.FeatureGroup(name="Travel time to nearest store", show=True)
    folium.GeoJson(
        blocks_wgs[["geometry"] + tooltip_fields],
        style_function=time_style,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
        ),
    ).add_to(time_layer)
    time_layer.add_to(m)
    time_colormap.add_to(m)

    # ------------------------------------------------------------------ #
    # Layer 2 — Store count (sequential Blues)                            #
    # ------------------------------------------------------------------ #
    vmax_count = max(int(blocks_wgs["store_count"].max()), 1)
    count_colormap = cm.LinearColormap(
        colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
        vmin=0,
        vmax=vmax_count,
        caption="Stores reachable within 30-min round-trip walk",
    )

    def count_style(feature):
        val = feature["properties"].get("store_count", 0)
        color = "#dddddd" if val == 0 else count_colormap(min(int(val), vmax_count))
        return {"fillColor": color, "color": "#444444", "weight": 0.2, "fillOpacity": 0.75}

    count_layer = folium.FeatureGroup(name="Stores reachable within 30 min", show=False)
    folium.GeoJson(
        blocks_wgs[["geometry"] + tooltip_fields],
        style_function=count_style,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
        ),
    ).add_to(count_layer)
    count_layer.add_to(m)
    count_colormap.add_to(m)

    # ------------------------------------------------------------------ #
    # Store markers (shared)                                              #
    # ------------------------------------------------------------------ #
    store_name_col = next(
        (c for c in retailer_gdf.columns if c.lower() in ("store_name", "name")), None
    )
    stores_layer = folium.FeatureGroup(name="Grocery stores", show=True)
    for _, row in stores_wgs.iterrows():
        label = str(row[store_name_col]) if store_name_col else "Grocery store"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            popup=folium.Popup(label, max_width=250),
            tooltip=label,
        ).add_to(stores_layer)
    stores_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    print(f"Block map saved to {output_path}")


# ---------------------------------------------------------------------------
# Step 10b — Interactive tract-level map: access thresholds + poverty
# ---------------------------------------------------------------------------

def make_tract_map(tract_gdf, retailer_gdf, output_path=OUTPUT_TRACT_MAP):
    """
    Interactive Folium choropleth at Census-tract resolution.

    Six toggleable layers (one visible at a time via radio-style layer control):
      - % population with access to ≥1 store  (default visible)
      - % population with access to ≥2 stores
      - % population with access to ≥3 stores
      - % population with access to ≥4 stores
      - % population with access to ≥5 stores
      - % population below the federal poverty line

    Access layers use a green→red scale (high access = green).
    Poverty layer uses an orange→red scale (high poverty = dark red).
    Hover a tract to see all metrics at once.
    Grocery stores are plotted as circle markers on all layers.
    """
    import folium
    import branca.colormap as cm

    tracts_wgs = tract_gdf.to_crs(4326).copy()
    stores_wgs = retailer_gdf.to_crs(4326)

    center = [
        tracts_wgs.geometry.centroid.y.mean(),
        tracts_wgs.geometry.centroid.x.mean(),
    ]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # Shared green→red colormap for access layers (0–100%)
    access_colormap = cm.LinearColormap(
        colors=["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
        vmin=0, vmax=100,
        caption="% population with food access",
    )

    # Orange→dark-red colormap for poverty (0–50%)
    poverty_vmax = max(float(tracts_wgs["pct_poverty"].quantile(0.95)), 20.0) \
        if "pct_poverty" in tracts_wgs.columns else 50.0
    poverty_colormap = cm.LinearColormap(
        colors=["#feedde", "#fdbe85", "#fd8d3c", "#e6550d", "#a63603"],
        vmin=0, vmax=poverty_vmax,
        caption="% population below 200% federal poverty level",
    )

    # Tooltip fields shown on hover for every layer
    access_cols = [f"pct_access_{n}store" for n in range(1, 6)]
    hover_fields = ["TRACT_GEOID", "total_pop"] + access_cols
    hover_aliases = ["Tract GEOID", "Population"] + [f"≥{n} store(s) (%)" for n in range(1, 6)]
    if "pct_poverty" in tracts_wgs.columns:
        hover_fields.append("pct_poverty")
        hover_aliases.append("Below 200% FPL (%)")

    # Round display columns
    for col in access_cols + (["pct_poverty"] if "pct_poverty" in tracts_wgs.columns else []):
        tracts_wgs[col] = tracts_wgs[col].round(1)

    def make_tooltip():
        """Each GeoJson layer needs its own tooltip instance (Folium single-parent rule)."""
        return folium.GeoJsonTooltip(
            fields=hover_fields,
            aliases=hover_aliases,
            localize=True,
            sticky=True,
        )

    # ------------------------------------------------------------------ #
    # Access layers (≥1 … ≥5 stores)                                      #
    # ------------------------------------------------------------------ #
    for i, n in enumerate(range(1, 6)):
        col = f"pct_access_{n}store"

        def make_style(column):
            def style_fn(feature):
                val = feature["properties"].get(column)
                color = "#cccccc" if val is None else access_colormap(min(float(val), 100))
                return {"fillColor": color, "color": "#555555", "weight": 0.4, "fillOpacity": 0.75}
            return style_fn

        layer = folium.FeatureGroup(
            name=f"Access: ≥{n} store{'s' if n > 1 else ''} within 30 min",
            show=(n == 1),   # only ≥1 store visible by default
        )
        folium.GeoJson(
            tracts_wgs[["geometry"] + hover_fields],
            style_function=make_style(col),
            tooltip=make_tooltip(),
        ).add_to(layer)
        layer.add_to(m)

    access_colormap.caption = "% population with access (green = more access)"
    access_colormap.add_to(m)

    # ------------------------------------------------------------------ #
    # Poverty layer                                                        #
    # ------------------------------------------------------------------ #
    if "pct_poverty" in tracts_wgs.columns:
        def poverty_style(feature):
            val = feature["properties"].get("pct_poverty")
            color = "#cccccc" if val is None else poverty_colormap(min(float(val), poverty_vmax))
            return {"fillColor": color, "color": "#555555", "weight": 0.4, "fillOpacity": 0.75}

        poverty_layer = folium.FeatureGroup(name="Poverty rate (%)", show=False)
        folium.GeoJson(
            tracts_wgs[["geometry"] + hover_fields],
            style_function=poverty_style,
            tooltip=make_tooltip(),
        ).add_to(poverty_layer)
        poverty_layer.add_to(m)
        poverty_colormap.add_to(m)

    # ------------------------------------------------------------------ #
    # Bivariate layers — low access × high poverty (one per threshold)    #
    # ------------------------------------------------------------------ #
    if "pct_poverty" in tracts_wgs.columns:
        # 3×3 color matrix: rows = access tercile (1=low … 3=high)
        #                   cols = poverty tercile (1=low … 3=high)
        # Palette: blue axis = more access, red axis = more poverty,
        # dark purple = low access + high poverty (the "concern" cell).
        BIVARIATE_COLORS = {
            (1, 1): "#e8e8e8",  # low access,  low poverty
            (1, 2): "#dfb0d6",  # low access,  mid poverty
            (1, 3): "#be64ac",  # low access,  high poverty  ← concern
            (2, 1): "#ace4e4",  # mid access,  low poverty
            (2, 2): "#a5b4c2",  # mid access,  mid poverty
            (2, 3): "#8c62aa",  # mid access,  high poverty
            (3, 1): "#5ac8c8",  # high access, low poverty
            (3, 2): "#5698b9",  # high access, mid poverty
            (3, 3): "#3b4994",  # high access, high poverty
        }

        # Poverty breaks are the same for all bivariate layers
        def _safe_breaks(series):
            """Quantile tercile breaks; falls back to equal-width if data collapses."""
            q1, q2 = series.quantile(0.333), series.quantile(0.667)
            if q1 == q2:
                lo, hi = series.min(), series.max()
                span = hi - lo if hi > lo else 100
                return [lo + span / 3, lo + 2 * span / 3]
            return [q1, q2]

        def _bin(val, breaks):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return 1 if val <= breaks[0] else (2 if val <= breaks[1] else 3)

        p_breaks = _safe_breaks(tracts_wgs["pct_poverty"].dropna())

        for bv_n in range(1, 6):
            bv_col = f"pct_access_{bv_n}store"
            a_breaks = _safe_breaks(tracts_wgs[bv_col].dropna())

            color_col = f"_bv_color_{bv_n}"
            tracts_wgs[color_col] = tracts_wgs.apply(
                lambda row, ac=bv_col, ab=a_breaks: BIVARIATE_COLORS.get(
                    (_bin(row[ac], ab), _bin(row["pct_poverty"], p_breaks)), "#cccccc"
                ),
                axis=1,
            )

            def make_bv_style(cc):
                def bv_style(feature):
                    color = feature["properties"].get(cc, "#cccccc")
                    return {"fillColor": color, "color": "#555555",
                            "weight": 0.4, "fillOpacity": 0.8}
                return bv_style

            bv_layer = folium.FeatureGroup(
                name=f"Bivariate: ≥{bv_n} store{'s' if bv_n > 1 else ''} access + poverty",
                show=False,
            )
            folium.GeoJson(
                tracts_wgs[["geometry"] + hover_fields + [color_col]],
                style_function=make_bv_style(color_col),
                tooltip=make_tooltip(),
            ).add_to(bv_layer)
            bv_layer.add_to(m)

        # Single shared legend (same color matrix regardless of threshold)
        bv_legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:10px;border:1px solid #ccc;
                    border-radius:4px;font-size:11px;line-height:1.4;">
          <b>Bivariate: access + poverty</b><br>
          <span style="font-size:9px;color:#444;">select a bivariate layer above</span><br>
          <table style="border-collapse:collapse;margin-top:4px;">
            <tr>
              <td style="font-size:10px;writing-mode:vertical-rl;
                         transform:rotate(180deg);text-align:center;
                         padding-right:3px;" rowspan="3">&#8593; more access</td>
              <td style="width:18px;height:18px;background:#5ac8c8;"></td>
              <td style="width:18px;height:18px;background:#5698b9;"></td>
              <td style="width:18px;height:18px;background:#3b4994;"></td>
            </tr>
            <tr>
              <td style="width:18px;height:18px;background:#ace4e4;"></td>
              <td style="width:18px;height:18px;background:#a5b4c2;"></td>
              <td style="width:18px;height:18px;background:#8c62aa;"></td>
            </tr>
            <tr>
              <td style="width:18px;height:18px;background:#e8e8e8;"></td>
              <td style="width:18px;height:18px;background:#dfb0d6;"></td>
              <td style="width:18px;height:18px;background:#be64ac;
                         outline:2px solid #333;"></td>
            </tr>
            <tr>
              <td></td>
              <td colspan="3" style="font-size:10px;text-align:center;
                                     padding-top:2px;">more poverty &#8594;</td>
            </tr>
          </table>
          <span style="font-size:9px;color:#666;">bins = terciles &nbsp;&#9632; = concern</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(bv_legend_html))

    # ------------------------------------------------------------------ #
    # Store markers                                                        #
    # ------------------------------------------------------------------ #
    store_name_col = next(
        (c for c in retailer_gdf.columns if c.lower() in ("store_name", "name")), None
    )
    stores_layer = folium.FeatureGroup(name="Grocery stores", show=True)
    for _, row in stores_wgs.iterrows():
        label = str(row[store_name_col]) if store_name_col else "Grocery store"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="black",
            fill=True,
            fill_color="white",
            fill_opacity=0.9,
            popup=folium.Popup(label, max_width=250),
            tooltip=label,
        ).add_to(stores_layer)
    stores_layer.add_to(m)

    # ------------------------------------------------------------------ #
    # Vulnerability score layers — (1 - access) × poverty (one per N)    #
    # ------------------------------------------------------------------ #
    # score = (1 - pct_access_Nstore/100) × (pct_poverty/100)
    # Range 0–1: high = low access AND high poverty = most concern.
    if "pct_poverty" in tracts_wgs.columns:
        vuln_vmax = 0.0
        for n in range(1, 6):
            col = f"_vuln_{n}"
            tracts_wgs[col] = (
                (1 - tracts_wgs[f"pct_access_{n}store"] / 100)
                * (tracts_wgs["pct_poverty"] / 100)
            ).round(4)
            vuln_vmax = max(vuln_vmax, float(tracts_wgs[col].max()))

        vuln_colormap = cm.LinearColormap(
            colors=["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
            vmin=0,
            vmax=vuln_vmax,
            caption="Vulnerability score: (1 − access) × % below 200% FPL",
        )
        vuln_colormap.add_to(m)

        for n in range(1, 6):
            col = f"_vuln_{n}"

            def make_vuln_style(c, vmax):
                def style_fn(feature):
                    val = feature["properties"].get(c)
                    color = "#cccccc" if val is None else vuln_colormap(min(float(val), vmax))
                    return {"fillColor": color, "color": "#555555",
                            "weight": 0.4, "fillOpacity": 0.8}
                return style_fn

            vuln_layer = folium.FeatureGroup(
                name=f"Vulnerability: ≥{n} store{'s' if n > 1 else ''} access × poverty",
                show=False,
            )
            folium.GeoJson(
                tracts_wgs[["geometry"] + hover_fields + [col]],
                style_function=make_vuln_style(col, vuln_vmax),
                tooltip=make_tooltip(),
            ).add_to(vuln_layer)
            vuln_layer.add_to(m)

    # ------------------------------------------------------------------ #
    # Priority population layers — people in poverty without access       #
    # ------------------------------------------------------------------ #
    # priority_pop = total_pop × (pct_poverty/100) × (1 - pct_access/100)
    # An estimated headcount of residents who are both in poverty and
    # lack access to N stores. Directly actionable: rank tracts by people
    # who need help, not by a dimensionless score.
    if "pct_poverty" in tracts_wgs.columns:
        priority_vmax = 0
        for n in range(1, 6):
            col = f"_priority_{n}"
            tracts_wgs[col] = (
                tracts_wgs["total_pop"]
                * (tracts_wgs["pct_poverty"] / 100)
                * (1 - tracts_wgs[f"pct_access_{n}store"] / 100)
            ).round(0).fillna(0).astype(int)
            priority_vmax = max(priority_vmax, int(tracts_wgs[col].max()))

        priority_colormap = cm.LinearColormap(
            colors=["#f7f4f9", "#d4b9da", "#df65b0", "#dd1c77", "#67001f"],
            vmin=0,
            vmax=priority_vmax,
            caption="Est. residents below 200% FPL without food access (headcount)",
        )
        priority_colormap.add_to(m)

        # Add priority_pop columns to hover tooltip
        priority_cols = [f"_priority_{n}" for n in range(1, 6)]
        priority_hover = hover_fields + priority_cols
        priority_aliases = hover_aliases + [
            f"<200% FPL, no ≥{n} store{'s' if n > 1 else ''} access" for n in range(1, 6)
        ]

        def make_priority_tooltip():
            return folium.GeoJsonTooltip(
                fields=priority_hover,
                aliases=priority_aliases,
                localize=True,
                sticky=True,
            )

        for n in range(1, 6):
            col = f"_priority_{n}"

            def make_priority_style(c, vmax):
                def style_fn(feature):
                    val = feature["properties"].get(c, 0)
                    color = "#cccccc" if val is None else priority_colormap(min(float(val), vmax))
                    return {"fillColor": color, "color": "#555555",
                            "weight": 0.4, "fillOpacity": 0.8}
                return style_fn

            priority_layer = folium.FeatureGroup(
                name=f"Priority pop: ≥{n} store{'s' if n > 1 else ''} access + poverty",
                show=False,
            )
            folium.GeoJson(
                tracts_wgs[["geometry"] + priority_hover],
                style_function=make_priority_style(col, priority_vmax),
                tooltip=make_priority_tooltip(),
            ).add_to(priority_layer)
            priority_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_path)
    print(f"Tract map saved to {output_path}")


# ---------------------------------------------------------------------------
# Priority map — beautiful standalone web app
# ---------------------------------------------------------------------------

def make_priority_map(tract_gdf, retailer_gdf, blocks_gdf=None,
                      output_path=OUTPUT_PRIORITY_MAP):
    """
    Standalone priority map: residents below 200% FPL with <2 store access.

    metric = total_pop × (pct_poverty/100) × (1 − pct_access_2store/100)

    Features:
      - Tracts clipped to SF land boundary (no ocean bleed)
      - Dark basemap, dark-navy → bright-red color scale
      - Fixed sidebar: city total, top-20 ranked tracts, click-to-fly
      - Block drill-down: clicking a tract shows per-block circles colored
        by store_count (red=0, orange=1, green=2+) with travel-time tooltip
    """
    import folium
    import json

    # -- 1. Tracts already use cartographic boundaries (coastline-clipped) ---
    tracts = tract_gdf.to_crs(4326).copy()
    print(f"  Using cartographic tract boundaries ({len(tracts)} tracts)")

    # -- 2. Core metric ----------------------------------------------------
    tracts["priority_pop"] = (
        tracts["total_pop"]
        * (tracts["pct_poverty"] / 100)
        * (1 - tracts["pct_access_2store"] / 100)
    ).round(0).fillna(0).astype(int)
    tracts["priority_rank"] = (
        tracts["priority_pop"].rank(ascending=False, method="min").astype(int)
    )
    tracts["pct_poverty"]     = tracts["pct_poverty"].round(1)
    tracts["pct_access_2store"] = tracts["pct_access_2store"].round(1)

    city_total = int(tracts["priority_pop"].sum())
    vmax       = max(int(tracts["priority_pop"].quantile(0.98)), 1)

    # -- 3. Color scale: YlOrBr sequential (colorblind-safe, low=light, high=dark) -
    # All tracts get a color — light cream for low/zero priority, dark brown-red for high.
    COLORS = ["#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]

    def heat_color(val, _vmax=vmax):
        frac  = min(float(val) / _vmax, 1.0) if (val and val > 0) else 0.0
        stops = [(i / (len(COLORS) - 1), c) for i, c in enumerate(COLORS)]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]; t1, c1 = stops[i + 1]
            if t0 <= frac <= t1:
                t = (frac - t0) / (t1 - t0)
                r = int(int(c0[1:3], 16) * (1-t) + int(c1[1:3], 16) * t)
                g = int(int(c0[3:5], 16) * (1-t) + int(c1[3:5], 16) * t)
                b = int(int(c0[5:7], 16) * (1-t) + int(c1[5:7], 16) * t)
                return f"#{r:02x}{g:02x}{b:02x}"
        return COLORS[-1]

    tracts["_color"] = tracts["priority_pop"].apply(heat_color)

    # -- 4. Block drill-down data (polygon GeoJSON keyed by tract) -----------
    block_data = {}   # {TRACT_GEOID: GeoJSON FeatureCollection}
    if blocks_gdf is not None:
        import json as _json
        blks = blocks_gdf.to_crs(4326).copy()
        tract_features = {}
        for _, row in blks.iterrows():
            tid = row.get("TRACT_GEOID")
            if not tid:
                continue
            rt  = row.get("min_rt_min")
            rt_val = round(float(rt), 1) if (rt is not None and rt == rt) else None
            sc  = int(row.get("store_count", 0))
            pop = int(row.get("POP100", 0))
            from shapely.geometry import mapping as _mapping
            feat = {
                "type": "Feature",
                "geometry": _mapping(row.geometry),
                "properties": {"rt": rt_val, "sc": sc, "pop": pop},
            }
            tract_features.setdefault(tid, []).append(feat)
        for tid, feats in tract_features.items():
            block_data[tid] = {"type": "FeatureCollection", "features": feats}

    # -- 5. Build Folium map -----------------------------------------------
    center  = [37.764, -122.437]
    m       = folium.Map(location=center, zoom_start=13,
                         tiles="CartoDB dark_matter", prefer_canvas=True)
    map_var = m.get_name()

    def tract_style(feature):
        return {"fillColor": feature["properties"].get("_color", "#1e293b"),
                "color": "#1e3a5f", "weight": 0.5, "fillOpacity": 0.82}

    def tract_highlight(feature):
        return {"color": "#fc8d59", "weight": 2, "fillOpacity": 0.92}

    display_cols = ["TRACT_GEOID", "priority_pop", "priority_rank",
                    "total_pop", "pct_poverty", "pct_access_2store", "_color"]
    gj = folium.GeoJson(
        tracts[display_cols + ["geometry"]],
        style_function=tract_style,
        highlight_function=tract_highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=["TRACT_GEOID", "priority_pop", "priority_rank",
                    "total_pop", "pct_poverty", "pct_access_2store"],
            aliases=["Tract", "Priority pop", "City rank",
                     "Total pop", "Below 200% FPL (%)", "≥2 store access (%)"],
            localize=True, sticky=True,
            style=("background:#0f172a;color:#f1f5f9;border:1px solid #1e3a5f;"
                   "border-radius:6px;font-family:system-ui,sans-serif;"
                   "font-size:12px;padding:10px 14px;"),
        ),
        name="tracts",
    )
    gj.add_to(m)
    gj_var = gj.get_name()

    # Store markers — in a FeatureGroup so we can hide on drill-down
    store_fg = folium.FeatureGroup(name="stores", show=True)
    stores_wgs = retailer_gdf.to_crs(4326)
    store_name_col = next(
        (c for c in retailer_gdf.columns if c.lower() in ("store_name", "name")), None
    )
    for _, row in stores_wgs.iterrows():
        label = str(row[store_name_col]) if store_name_col else "Grocery store"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6, color="#1a1a1a", fill=True,
            fill_color="#ffffff", fill_opacity=0.95, weight=1.5,
            tooltip=label,
        ).add_to(store_fg)
    store_fg.add_to(m)
    store_fg_var = store_fg.get_name()

    # -- 6. Sidebar HTML ---------------------------------------------------
    top20 = tracts.nlargest(20, "priority_pop")
    tract_items = []
    max_pop = max(int(top20["priority_pop"].iloc[0]), 1)
    for _, row in top20.iterrows():
        geoid   = row["TRACT_GEOID"]
        rank    = int(row["priority_rank"])
        pop     = int(row["priority_pop"])
        poverty = float(row["pct_poverty"]) if row["pct_poverty"] == row["pct_poverty"] else 0
        access  = float(row["pct_access_2store"]) if row["pct_access_2store"] == row["pct_access_2store"] else 0
        color   = row["_color"]
        bar_pct = min(int(pop / max_pop * 100), 100)
        has_blocks = "true" if geoid in block_data else "false"

        tract_items.append(f"""
        <div class="tract-item" data-geoid="{geoid}" data-has-blocks="{has_blocks}"
             onclick="selectTract('{geoid}')">
          <div class="tract-header">
            <div>
              <span class="rank">#{rank}</span>
              <span class="tract-id">Tract {geoid[-6:]}</span>
            </div>
            <span class="pop-num">{pop:,}</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{bar_pct}%;background:{color};"></div>
          </div>
          <div class="tract-meta">
            {poverty:.0f}% below 200% FPL &nbsp;&middot;&nbsp; {access:.0f}% reach 2+ stores
          </div>
        </div>""")

    centroids = {
        row["TRACT_GEOID"]: [row.geometry.centroid.x, row.geometry.centroid.y]
        for _, row in tracts.iterrows()
    }

    sidebar_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      #priority-sidebar {{
        position:fixed; top:0; left:0; width:310px; height:100vh;
        background:#0a1120; color:#f1f5f9; z-index:9999;
        display:flex; flex-direction:column;
        font-family:'Inter',system-ui,sans-serif;
        box-shadow:4px 0 24px rgba(0,0,0,0.7);
      }}
      #sidebar-header {{ padding:22px 18px 0; flex-shrink:0; }}
      .eyebrow {{ font-size:10px; letter-spacing:2px; color:#475569;
                  text-transform:uppercase; margin-bottom:6px; }}
      #sidebar-header h1 {{ font-size:17px; font-weight:700; margin:0 0 5px;
                            line-height:1.25; color:#f8fafc; }}
      #sidebar-header p {{ font-size:11px; color:#64748b; margin:0 0 14px;
                           line-height:1.5; }}
      .stat-box {{ background:#1a2540; border-radius:8px; padding:13px 16px;
                   margin-bottom:14px; border:1px solid #1e3a5f; }}
      .stat-label {{ font-size:10px; color:#64748b; letter-spacing:1px;
                     text-transform:uppercase; margin-bottom:3px; }}
      .stat-value {{ font-size:30px; font-weight:700; color:#e34a33; line-height:1; }}
      .stat-sub {{ font-size:10px; color:#475569; margin-top:2px; }}
      .section-label {{ font-size:10px; letter-spacing:1px; color:#475569;
                        text-transform:uppercase; padding:0 18px 8px; flex-shrink:0; }}
      #main-list {{ overflow-y:auto; flex:1; padding:0 10px 16px; }}
      #main-list::-webkit-scrollbar {{ width:3px; }}
      #main-list::-webkit-scrollbar-thumb {{ background:#1e3a5f; border-radius:2px; }}
      .tract-item {{ background:#111827; border-radius:7px; padding:10px 12px;
                     margin-bottom:5px; cursor:pointer; border:1px solid #1e293b;
                     transition:border-color 0.12s, background 0.12s; }}
      .tract-item:hover {{ background:#1a2540; border-color:#1e3a5f; }}
      .tract-item.active {{ background:#1a2540; border-color:#e34a33; }}
      .tract-header {{ display:flex; justify-content:space-between;
                       align-items:center; margin-bottom:5px; }}
      .rank {{ font-size:10px; color:#475569; }}
      .tract-id {{ font-size:13px; font-weight:600; margin-left:4px; color:#e2e8f0; }}
      .pop-num {{ font-size:14px; font-weight:700; color:#e34a33; }}
      .bar-bg {{ height:2px; background:#1e293b; border-radius:2px; margin-bottom:5px; }}
      .bar-fill {{ height:2px; border-radius:2px; }}
      .tract-meta {{ font-size:10px; color:#475569; }}
      /* Drill-down panel */
      #drill-panel {{ display:none; flex-direction:column; flex:1; overflow:hidden; }}
      #drill-back {{ display:flex; align-items:center; gap:6px; padding:10px 18px;
                     cursor:pointer; color:#64748b; font-size:11px;
                     border-bottom:1px solid #1e293b; flex-shrink:0; }}
      #drill-back:hover {{ color:#e2e8f0; }}
      #drill-title {{ padding:12px 18px 6px; flex-shrink:0; }}
      #drill-title .tract-id {{ font-size:14px; color:#f8fafc; }}
      #drill-stats {{ padding:0 18px 10px; flex-shrink:0; }}
      #drill-stats .ds {{ font-size:11px; color:#64748b; margin-bottom:3px; }}
      #drill-stats .ds span {{ color:#e2e8f0; font-weight:600; }}
      #block-list {{ overflow-y:auto; flex:1; padding:0 10px 16px; }}
      #block-list::-webkit-scrollbar {{ width:3px; }}
      #block-list::-webkit-scrollbar-thumb {{ background:#1e293b; border-radius:2px; }}
      .block-item {{ background:#111827; border-radius:5px; padding:7px 10px;
                     margin-bottom:4px; border-left:3px solid #334155; }}
      .block-row {{ display:flex; justify-content:space-between; align-items:center; }}
      .block-sc {{ font-size:12px; font-weight:600; }}
      .block-meta {{ font-size:10px; color:#475569; margin-top:2px; }}
      /* Legend */
      .legend-wrap {{ flex-shrink:0; padding:12px 18px;
                      border-top:1px solid #1e293b; }}
      .legend-label {{ font-size:10px; color:#475569; margin-bottom:5px; }}
      .legend-bar {{ height:7px; border-radius:3px;
                     background:linear-gradient(to right,#fff7ec,#fdd49e,#fc8d59,#e34a33,#b30000);
                     margin-bottom:3px; }}
      .legend-ends {{ display:flex; justify-content:space-between;
                      font-size:10px; color:#475569; }}
      /* Methods button */
      #methods-btn {{
        position:fixed; bottom:20px; right:20px; z-index:10000;
        background:#1a2540; border:1px solid #1e3a5f; color:#94a3b8;
        border-radius:50%; width:38px; height:38px;
        font-size:16px; font-weight:700; cursor:pointer;
        display:flex; align-items:center; justify-content:center;
        transition:background 0.15s, color 0.15s;
        font-family:'Inter',system-ui,sans-serif;
      }}
      #methods-btn:hover {{ background:#1e3a5f; color:#f1f5f9; }}
      /* Methods modal */
      #methods-modal {{
        display:none; position:fixed; inset:0; z-index:10001;
        background:rgba(0,0,0,0.65); align-items:center; justify-content:center;
      }}
      #methods-modal.open {{ display:flex; }}
      #methods-box {{
        background:#0f1a2e; border:1px solid #1e3a5f; border-radius:12px;
        width:min(560px,90vw); max-height:80vh; overflow-y:auto;
        padding:28px 30px; color:#cbd5e1;
        font-family:'Inter',system-ui,sans-serif; font-size:13px; line-height:1.7;
        box-shadow:0 8px 40px rgba(0,0,0,0.7);
      }}
      #methods-box::-webkit-scrollbar {{ width:4px; }}
      #methods-box::-webkit-scrollbar-thumb {{ background:#1e3a5f; border-radius:2px; }}
      #methods-box h2 {{ font-size:16px; font-weight:700; color:#f8fafc;
                         margin:0 0 4px; }}
      #methods-box .sub {{ font-size:10px; color:#475569; letter-spacing:1px;
                           text-transform:uppercase; margin-bottom:20px; }}
      #methods-box h3 {{ font-size:12px; font-weight:600; color:#94a3b8;
                         text-transform:uppercase; letter-spacing:1px;
                         margin:20px 0 6px; border-top:1px solid #1e293b;
                         padding-top:14px; }}
      #methods-box p {{ margin:0 0 10px; }}
      #methods-box code {{ background:#1e293b; border-radius:3px;
                           padding:1px 5px; font-size:11px; color:#7dd3fc; }}
      #methods-box a {{ color:#7dd3fc; text-decoration:none; }}
      #methods-close {{
        float:right; cursor:pointer; color:#475569; font-size:18px;
        line-height:1; margin-left:12px;
      }}
      #methods-close:hover {{ color:#f1f5f9; }}
    </style>

    <div id="priority-sidebar">
      <!-- Main list view -->
      <div id="main-view" style="display:flex;flex-direction:column;flex:1;overflow:hidden;">
        <div id="sidebar-header">
          <div class="eyebrow">SF Food Bank · 2026</div>
          <h1>Food Access<br>Priority Map</h1>
          <p>Residents below 200% FPL with access<br>to fewer than 2 stores within 30 min.</p>
          <div class="stat-box">
            <div class="stat-label">City-wide priority population</div>
            <div class="stat-value">{city_total:,}</div>
            <div class="stat-sub">estimated residents</div>
          </div>
        </div>
        <div class="section-label">Top tracts by priority population</div>
        <div id="main-list">{''.join(tract_items)}</div>
        <div class="legend-wrap">
          <div class="legend-label">Priority population intensity</div>
          <div class="legend-bar"></div>
          <div class="legend-ends"><span>Low / full access</span><span>High ({vmax:,}+)</span></div>
        </div>
      </div>

      <!-- Drill-down view (hidden until tract selected) -->
      <div id="drill-panel">
        <div id="drill-back" onclick="closeDrill()">&#8592; Back to all tracts</div>
        <div id="drill-title"><span class="tract-id" id="drill-tract-id"></span></div>
        <div id="drill-stats">
          <div class="ds">Priority population: <span id="ds-priority"></span></div>
          <div class="ds">Below 200% FPL: <span id="ds-poverty"></span></div>
          <div class="ds">Reach 2+ stores: <span id="ds-access"></span></div>
          <div class="ds" style="margin-top:6px;font-size:10px;color:#334155;">
            Blocks colored by round-trip travel time.<br>
            Block population = <em>all residents</em> (2020 Census), not just those below 200% FPL &mdash; income data is only available at the tract level.
          </div>
        </div>
        <div style="padding:0 18px 8px;flex-shrink:0;">
          <div style="font-size:10px;color:#475569;margin-bottom:3px;">Round-trip walk time</div>
          <div style="height:6px;border-radius:3px;background:linear-gradient(to right,#fff7ec,#fdd49e,#fc8d59,#d7301f,#7f0000);margin-bottom:2px;"></div>
          <div style="display:flex;justify-content:space-between;font-size:9px;color:#475569;">
            <span>0 min</span><span>15 min</span><span>30+ min</span>
          </div>
          <div style="margin-top:5px;font-size:9px;color:#334155;">Grey = unreachable (&gt;30 min)</div>
        </div>
        <div id="block-list"></div>
      </div>
    </div>

    <!-- Methods button -->
    <button id="methods-btn" onclick="document.getElementById('methods-modal').classList.add('open')" title="Methods">?</button>

    <!-- Methods modal -->
    <div id="methods-modal" onclick="if(event.target===this)this.classList.remove('open')">
      <div id="methods-box">
        <span id="methods-close" onclick="document.getElementById('methods-modal').classList.remove('open')">&times;</span>
        <h2>About This Map</h2>
        <div class="sub">SF Food Bank &middot; Food Access Analysis &middot; 2026</div>

        <!-- Tab buttons -->
        <div style="display:flex;gap:8px;margin-bottom:18px;">
          <button id="tab-summary" onclick="switchTab('summary')"
            style="flex:1;padding:7px 0;border-radius:6px;border:1px solid #1e3a5f;
                   background:#1a2540;color:#f1f5f9;font-size:12px;font-weight:600;
                   cursor:pointer;font-family:inherit;">Overview</button>
          <button id="tab-technical" onclick="switchTab('technical')"
            style="flex:1;padding:7px 0;border-radius:6px;border:1px solid #1e293b;
                   background:transparent;color:#64748b;font-size:12px;font-weight:600;
                   cursor:pointer;font-family:inherit;">Technical Details</button>
        </div>

        <!-- Summary tab -->
        <div id="content-summary">
          <h3>What does this map show?</h3>
          <p>This map highlights the San Francisco neighborhoods where <strong>low-income residents have the hardest time reaching a grocery store on foot.</strong> Darker-colored areas have more people who both struggle financially and lack nearby stores.</p>

          <h3>How is "low income" defined?</h3>
          <p>We use the federal poverty line as a benchmark. A family of four earning less than about <strong>$60,000 per year</strong> falls below 200% of the poverty level &mdash; a common threshold used to identify food insecurity risk. Estimates come from the U.S. Census Bureau's American Community Survey (2022).</p>

          <h3>How is "access" measured?</h3>
          <p>We measure whether someone can <strong>walk to a grocery store and back in 30 minutes or less.</strong> This accounts for San Francisco's hills &mdash; steep climbs slow people down, so a short distance on a steep street can take longer than a longer flat route. A full grocery trip is also assumed to be slower on the way back due to carrying bags.</p>
          <p>A neighborhood scores poorly if fewer than 2 grocery stores are reachable within that 30-minute window &mdash; having only one store means no backup if it's closed, crowded, or unaffordable.</p>

          <h3>What are the colored blocks when I click a neighborhood?</h3>
          <p>Each small block is colored by how long the walk to the nearest store takes: <strong>cream = quick (&lt;10 min), orange = moderate, dark red = long (25&ndash;30 min), grey = no store reachable within 30 min.</strong></p>
          <p>The population count shown for each block is <strong>all residents</strong> as of the 2020 Decennial Census &mdash; not just those below 200% FPL. Income data at that level of detail isn't available from the Census; poverty estimates are only reliable at the neighborhood (tract) level.</p>

          <h3>What does the priority number mean?</h3>
          <p>The "priority population" is an estimate of the number of residents in a neighborhood who are both low-income <em>and</em> lack easy grocery access. It helps identify where food bank outreach or mobile pantry routes could have the most impact.</p>
        </div>

        <!-- Technical tab (hidden by default) -->
        <div id="content-technical" style="display:none;">
          <h3>Priority Metric</h3>
          <p>Priority population combines poverty rate and access gap at the Census tract level:</p>
          <p><code>priority_pop = total_pop &times; (pct_below_200%_FPL / 100) &times; (1 &minus; pct_access_2stores / 100)</code></p>
          <p>Poverty estimates use ACS 5-year 2022 table <strong>C17002</strong>, summing bands _002E&ndash;_007E (all ratios &lt;2.0) divided by _001E (universe). Measured at tract level only &mdash; block-level poverty is not available from ACS.</p>

          <h3>Tobler Hiking Function</h3>
          <p>Per-edge walk speed is adjusted for slope using the Tobler hiking function:</p>
          <p><code>v = 6 &times; exp(&minus;3.5 &times; |grade + 0.05|) km/h</code></p>
          <p>Edge grades are computed from the <strong>USGS 3DEP 10m DEM</strong> (fetched via py3dep). Each directed edge carries its own slope; the reverse edge carries the opposing slope. A <strong>1.2&times; load penalty</strong> is applied to return-trip edge weights.</p>

          <h3>Dual-Sweep Dijkstra</h3>
          <p>Instead of O(blocks &times; stores) routing, two shortest-path sweeps are run per store using the igraph C backend: an <em>inbound</em> sweep on the reversed graph (gives block&rarr;store costs) and an <em>outbound</em> sweep on the forward graph (gives store&rarr;block costs). Round-trip time = outbound + return. ~400 sweeps replace ~1.2M individual paths.</p>

          <h3>Network & Thresholds</h3>
          <p>Walk network: OSMnx pedestrian graph for San Francisco, projected to EPSG:32610. Threshold: <strong>1,800 seconds (30 min)</strong> round-trip. Retailers: USDA SNAP-authorized stores filtered to grocery/supermarket types via ArcGIS Hub REST API. Block geometries + POP100 from Census TIGERweb (2020 PL 94-171).</p>

          <h3>Data Sources</h3>
          <p>
            <a href="https://www.usda.gov/topics/food-and-nutrition/food-access" target="_blank">USDA SNAP retailers</a> (ArcGIS Hub) &middot;
            <a href="https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html" target="_blank">Census TIGERweb</a> &middot;
            <a href="https://www.openstreetmap.org" target="_blank">OpenStreetMap</a> &middot;
            <a href="https://www.usgs.gov/3d-elevation-program" target="_blank">USGS 3DEP</a>
          </p>
        </div>

      </div>
    </div>

    <script>
      var _centroids  = {json.dumps(centroids)};
      var _blockData  = {json.dumps(block_data)};
      var _drillLayer = null;
      var _activeItem = null;
      // Layer references resolved after Folium initializes (window load)
      var _tractLayer = null;
      var _storeLayer = null;
      var _mapObj     = null;

      // Interpolate between two hex colors
      function lerpColor(c0, c1, t) {{
        var r = Math.round(parseInt(c0.slice(1,3),16)*(1-t)+parseInt(c1.slice(1,3),16)*t);
        var g = Math.round(parseInt(c0.slice(3,5),16)*(1-t)+parseInt(c1.slice(3,5),16)*t);
        var b = Math.round(parseInt(c0.slice(5,7),16)*(1-t)+parseInt(c1.slice(5,7),16)*t);
        return '#'+[r,g,b].map(function(x){{return ('0'+x.toString(16)).slice(-2);}}).join('');
      }}

      // Color block by round-trip minutes — cream (fast/close) to dark mahogany (slow/far)
      // Single warm hue matches the tract palette; colorblind-safe (luminance-based)
      function rtColor(rt) {{
        if (rt === null || rt === undefined) return '#334155';
        var stops = [
          [0,  '#fff7ec'],
          [10, '#fdd49e'],
          [20, '#fc8d59'],
          [25, '#d7301f'],
          [30, '#7f0000'],
        ];
        var t = Math.min(rt, 30);
        for (var i = 0; i < stops.length - 1; i++) {{
          if (t <= stops[i+1][0]) {{
            var frac = (t - stops[i][0]) / (stops[i+1][0] - stops[i][0]);
            return lerpColor(stops[i][1], stops[i+1][1], frac);
          }}
        }}
        return stops[stops.length-1][1];
      }}

      function selectTract(geoid) {{
        var c = _centroids[geoid];
        if (c && _mapObj) {{ _mapObj.flyTo([c[1], c[0]], 15, {{duration: 1.0}}); }}

        if (_activeItem) _activeItem.classList.remove('active');
        _activeItem = document.querySelector('[data-geoid="' + geoid + '"]');
        if (_activeItem) _activeItem.classList.add('active');

        var fc = _blockData[geoid];
        if (fc && fc.features && fc.features.length > 0) {{
          showDrill(geoid, fc);
        }}
      }}

      function showDrill(geoid, fc) {{
        if (!_mapObj) return;
        if (_drillLayer) {{ _mapObj.removeLayer(_drillLayer); }}
        // Hide tract + store layers so blocks read clearly
        if (_tractLayer) _mapObj.removeLayer(_tractLayer);
        if (_storeLayer) _mapObj.removeLayer(_storeLayer);
        _drillLayer = L.layerGroup().addTo(_mapObj);

        // Render block polygons colored by travel time; grey out unpopulated blocks
        var blockGeoJSON = L.geoJSON(fc, {{
          style: function(feature) {{
            var p = feature.properties;
            var col = (p.pop === 0) ? '#475569' : rtColor(p.rt);
            var opacity = (p.pop === 0) ? 0.3 : 0.78;
            return {{
              fillColor: col,
              color: 'rgba(255,255,255,0.45)',
              weight: 1.2,
              fillOpacity: opacity,
            }};
          }},
          onEachFeature: function(feature, layer) {{
            var p = feature.properties;
            if (p.pop === 0) {{
              layer.bindTooltip('<i style="color:#94a3b8">Unpopulated block</i>', {{sticky: true}});
              return;
            }}
            var rtStr = p.rt !== null && p.rt !== undefined ? p.rt + ' min round trip' : 'Unreachable (>30 min)';
            layer.bindTooltip(
              '<b>' + rtStr + '</b><br>' +
              p.sc + ' store' + (p.sc !== 1 ? 's' : '') + ' reachable<br>' +
              'Pop: ' + p.pop,
              {{sticky: true}}
            );
          }}
        }});
        blockGeoJSON.addTo(_drillLayer);
        blockGeoJSON.bringToFront();

        // Update drill panel stats
        var item = document.querySelector('[data-geoid="' + geoid + '"]');
        var popNum = item ? item.querySelector('.pop-num').textContent : '';
        var meta   = item ? item.querySelector('.tract-meta').textContent.trim() : '';
        var parts  = meta.split('\u00b7').map(function(s){{return s.trim();}});

        document.getElementById('drill-tract-id').textContent = 'Tract ' + geoid.slice(-6);
        document.getElementById('ds-priority').textContent = popNum;
        document.getElementById('ds-poverty').textContent  = parts[0] ? parts[0].trim() : '';
        document.getElementById('ds-access').textContent   = parts[1] ? parts[1].trim() : '';

        // Build block list sorted by travel time
        var feats = fc.features.slice().sort(function(a,b) {{
          var ra = a.properties.rt !== null ? a.properties.rt : 999;
          var rb = b.properties.rt !== null ? b.properties.rt : 999;
          return ra - rb;
        }});
        var html = '';
        feats.forEach(function(f) {{
          var p = f.properties;
          var col = rtColor(p.rt);
          var rtStr = p.rt !== null && p.rt !== undefined ? p.rt + ' min round trip' : 'Unreachable';
          html += '<div class="block-item" style="border-left-color:' + col + '">' +
            '<div class="block-row">' +
            '<span class="block-sc" style="color:' + col + '">' + rtStr + '</span>' +
            '<span style="font-size:11px;color:#64748b;">pop ' + p.pop + '</span>' +
            '</div>' +
            '<div class="block-meta">' + p.sc + ' store' + (p.sc !== 1 ? 's' : '') + ' reachable</div>' +
            '</div>';
        }});
        document.getElementById('block-list').innerHTML = html;

        // Switch to drill view
        document.getElementById('main-view').style.display  = 'none';
        document.getElementById('drill-panel').style.display = 'flex';
      }}

      function closeDrill() {{
        if (!_mapObj) return;
        if (_drillLayer) {{ _mapObj.removeLayer(_drillLayer); _drillLayer = null; }}
        // Restore tract + store layers
        if (_tractLayer) _tractLayer.addTo(_mapObj);
        if (_storeLayer) _storeLayer.addTo(_mapObj);
        _mapObj.flyTo([37.764, -122.437], 13, {{duration: 0.8}});
        document.getElementById('drill-panel').style.display = 'none';
        document.getElementById('main-view').style.display   = 'flex';
        if (_activeItem) {{ _activeItem.classList.remove('active'); _activeItem = null; }}
      }}

      // Methods modal tab switcher
      function switchTab(tab) {{
        document.getElementById('content-summary').style.display   = tab === 'summary'   ? '' : 'none';
        document.getElementById('content-technical').style.display = tab === 'technical' ? '' : 'none';
        document.getElementById('tab-summary').style.background    = tab === 'summary'   ? '#1a2540' : 'transparent';
        document.getElementById('tab-summary').style.color         = tab === 'summary'   ? '#f1f5f9' : '#64748b';
        document.getElementById('tab-summary').style.borderColor   = tab === 'summary'   ? '#1e3a5f' : '#1e293b';
        document.getElementById('tab-technical').style.background  = tab === 'technical' ? '#1a2540' : 'transparent';
        document.getElementById('tab-technical').style.color       = tab === 'technical' ? '#f1f5f9' : '#64748b';
        document.getElementById('tab-technical').style.borderColor = tab === 'technical' ? '#1e3a5f' : '#1e293b';
      }}

      // Init after all Folium scripts have run
      window.addEventListener('load', function() {{
        _mapObj     = {map_var};
        _tractLayer = {gj_var};
        _storeLayer = {store_fg_var};

        // Map click on tract polygon
        _tractLayer.on('click', function(e) {{
          var geoid = e.layer.feature && e.layer.feature.properties
                      ? e.layer.feature.properties.TRACT_GEOID : null;
          if (geoid) selectTract(geoid);
        }});

        // Offset map for sidebar
        var mapEl = document.getElementById('{map_var}');
        if (mapEl) {{
          mapEl.style.marginLeft = '310px';
          mapEl.style.width = 'calc(100vw - 310px)';
          _mapObj.invalidateSize();
        }}
      }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(sidebar_html))
    m.save(output_path)
    print(f"Priority map saved to {output_path}  |  city-wide priority pop: {city_total:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Steps 1–3: graph + elevation + edge costs
    G = build_walk_graph(PLACE)
    graph_crs = ox.graph_to_gdfs(G, edges=False).crs

    if not os.path.exists(DEM_PATH):
        download_dem(G, DEM_PATH)
    else:
        print(f"Using existing DEM: {DEM_PATH}")
    G = add_tobler_costs(G, DEM_PATH)

    # Steps 4–5: input data
    retailers = load_retailers()
    blocks = load_census_blocks()

    # Step 6: snap to graph
    print("Snapping block centroids and stores to graph nodes ...")
    block_centroids = blocks.to_crs(graph_crs).copy()
    block_centroids["geometry"] = block_centroids.geometry.centroid
    block_nodes = snap_to_graph(G, block_centroids, graph_crs)
    store_nodes = snap_to_graph(G, retailers, graph_crs)
    unique_stores = list(dict.fromkeys(store_nodes))

    # Step 7: routing
    print(
        f"Computing round-trip times -- {len(unique_stores):,} store sweeps "
        f"-> {len(block_nodes):,} blocks ..."
    )
    min_times, store_counts = compute_min_roundtrip(G, block_nodes, store_nodes)

    blocks["min_rt_sec"] = min_times
    blocks["min_rt_min"] = blocks["min_rt_sec"].apply(
        lambda t: round(t / 60, 1) if t is not None else None
    )
    blocks["accessible"] = blocks["min_rt_sec"].apply(
        lambda t: t is not None and t <= THRESHOLD_SEC
    )
    blocks["store_count"] = store_counts

    n_accessible = blocks["accessible"].sum()
    print(f"Accessible blocks: {n_accessible:,} / {len(blocks):,}")

    # Step 8: tract rollup
    tracts = rollup_to_tracts(blocks)

    # Step 8b: merge poverty estimates
    poverty = load_tract_poverty()
    tracts = tracts.merge(poverty, on="TRACT_GEOID", how="left")

    tracts.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Tract results saved to {OUTPUT_GEOJSON}")

    # Cache blocks with routing results for priority map regeneration
    blocks_cache = os.path.join(OUTPUT_DIR, "_blocks_cache.gpkg")
    blocks.to_file(blocks_cache, driver="GPKG")
    print(f"Block routing cache saved to {blocks_cache}")

    # Step 9: tract-level static map (shows ≥1 store / pct_accessible)
    make_map(tracts, retailers)

    # Step 10: block-level interactive map
    make_block_map(blocks, retailers)

    # Step 10b: tract-level interactive map (access thresholds + poverty)
    make_tract_map(tracts, retailers)

    # Step 10c: standalone priority map
    make_priority_map(tracts, retailers, blocks_gdf=blocks)

    return tracts


if __name__ == "__main__":
    main()
