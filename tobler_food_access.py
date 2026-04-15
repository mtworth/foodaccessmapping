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
        print("  igraph not found — falling back to NetworkX (slower).")
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

def rollup_to_tracts(blocks_gdf):
    """
    Aggregate block-level accessibility to Census tracts.
    pct_accessible = accessible_population / total_population × 100.
    """
    tracts = (
        blocks_gdf.to_crs("EPSG:4326")
        .dissolve(by="TRACT_GEOID", aggfunc={"POP100": "sum"})
        .reset_index()
        .rename(columns={"POP100": "total_pop"})
    )

    accessible_pop = (
        blocks_gdf[blocks_gdf["accessible"]]
        .groupby("TRACT_GEOID")["POP100"]
        .sum()
        .reset_index()
        .rename(columns={"POP100": "accessible_pop"})
    )

    tracts = tracts.merge(accessible_pop, on="TRACT_GEOID", how="left")
    tracts["accessible_pop"] = tracts["accessible_pop"].fillna(0).astype(int)
    tracts["pct_accessible"] = np.where(
        tracts["total_pop"] > 0,
        tracts["accessible_pop"] / tracts["total_pop"] * 100,
        np.nan,
    )

    city_pop = tracts["total_pop"].sum()
    city_acc = tracts["accessible_pop"].sum()
    print(f"\nCity-wide: {city_acc:,} / {city_pop:,} people with food access "
          f"({city_acc / city_pop * 100:.1f}%)")
    print(f"Tracts >80% access : {(tracts['pct_accessible'] > 80).sum()}")
    print(f"Tracts <20% access : {(tracts['pct_accessible'] < 20).sum()}")

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
        f"Computing round-trip times — {len(unique_stores):,} store sweeps "
        f"→ {len(block_nodes):,} blocks ..."
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
    tracts.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Tract results saved to {OUTPUT_GEOJSON}")

    # Step 9: tract-level static map
    make_map(tracts, retailers)

    # Step 10: block-level interactive map
    make_block_map(blocks, retailers)

    return tracts


if __name__ == "__main__":
    main()
