# === Figure 2: Study Area and Köppen-Geiger Climate Classification ===
# Redesigned 2026-09-05 v12.20 (supersedes v11.6): per-station LCZ tags in c (right-first
# +3pt ring gap, twin-matched label sides), Fig-10a furniture (N-below arrow, plain 200 m bar),
# outer-ring-only campus boundary in b (c has none), tag/label halos tuned, no place labels,
# Koppen legend beside a on c rail, c title without citation (caption carries it).
# Prior note (v11.6): FULL-COLOR Koppen in a (author),
# a width-matched to b top-left with 6-class legend centered in the band to its right,
# zoom fan from the Georgia box to the twins OUTER corners at z0.5 (above a, below b/c
# so titles paint over it), spacer row for a tight panel-legend gap, dataviz-panel batch
# (swatch edges, dashed ring key, Cfa halo tag, slim cased Atlanta leader, label nudges,
# subtle white casing on c parcel dashes, slimmer c bar casing).
# Prior note (v10.9): a) Cfa-only muted
# Koppen locator TOP-LEFT with graticule; b/c campus twins with Fig-10a basemap language,
# 4pt cased nodes in dashed 100 m rings, block labels (b), place labels Midtown/Campus area,
# 250 m bars bottom-right, GT watermark bottom-left, compact centered legends per panel.
# Prior note (v7A): a) Cfa-only muted Koppen locator as
# top-center apex, b) land-cover and c) LCZ campus maps as equal same-window twins below,
# campus-tight crop, legends spread wide under their own panels, zoom connectors a->b and a->c.
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Shared publication config (canonical fonts, sizes, DPI)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_config import (FULL_W, DPI, FS_LABEL, FS_TICK, FS_DENSE, FS_DENSE_SM, inline_panel_title,
                         apply_rcparams, savefig)
apply_rcparams()

import socket
socket.setdefaulttimeout(30)  # 30s timeout for downloads
import osmnx as ox
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable decompression-bomb limit for Köppen global raster
import zipfile
import geopandas as gpd
import urllib.request
from matplotlib.patches import ConnectionPatch, Patch, Ellipse

# ── All ten study station coordinates (5 land cover types × 2) ──
STATIONS = {
    '10635576': {'name': '10th St Bridge',       'lat': 33.781539, 'lon': -84.391662, 'setting': 'Exposed'},
    '10635580': {'name': '5th St Bridge',        'lat': 33.776853, 'lon': -84.390820, 'setting': 'Exposed'},
    '10635581': {'name': 'Canopied Walkway',     'lat': 33.777870, 'lon': -84.400738, 'setting': 'Tree canopy'},
    '10635582': {'name': 'Hemphill Parking',     'lat': 33.781009, 'lon': -84.405840, 'setting': 'Impervious'},
    '10635585': {'name': 'Baker Building',       'lat': 33.780249, 'lon': -84.399261, 'setting': 'Built'},
    '10635587': {'name': 'Curran Parking Deck',  'lat': 33.778978, 'lon': -84.405711, 'setting': 'Impervious'},
    '10635588': {'name': 'Constructed Wetlands', 'lat': 33.781267, 'lon': -84.398088, 'setting': 'Open green'},
    '10635590': {'name': 'Grass Walkway',        'lat': 33.775932, 'lon': -84.400763, 'setting': 'Open green'},
    '10635593': {'name': 'Tech Square',          'lat': 33.776800, 'lon': -84.389269, 'setting': 'Built'},
    '10635595': {'name': 'Brittain Drive',       'lat': 33.774751, 'lon': -84.393105, 'setting': 'Tree canopy'},
}

# Marker styles by land cover type (5 types, colorblind-safe palette)
# fills from the paper-wide land-cover palette (fig_config.LC_COLORS), so the key learned here holds in Figs. 4, 7, 9, 10
# (2026-10-06; the earlier Okabe-Ito fills collided with the model colors LSTM/Attention-LSTM/DLinear); shapes unchanged
from fig_config import LC_COLORS as _LCC
SETTING_STYLES = {
    'Exposed':     {'marker': 's', 'color': _LCC['Exposed'],     'label': 'Exposed (bridge)'},
    'Tree canopy': {'marker': 'o', 'color': _LCC['Tree canopy'], 'label': 'Tree canopy'},
    'Impervious':  {'marker': 'P', 'color': _LCC['Impervious'],  'label': 'Impervious (parking)'},
    'Built':       {'marker': 'D', 'color': _LCC['Built'],       'label': 'Built (building)'},
    'Open green':  {'marker': 'v', 'color': _LCC['Open green'],  'label': 'Open green'},
}

# Atlanta center for panel (a) marker
ATLANTA_LAT, ATLANTA_LON = 33.778, -84.392

# ════════════════════════════════════════════════════════════════════
# DATA ACQUISITION (cached)
# ════════════════════════════════════════════════════════════════════
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

# --- 1. Beck et al. (2018) Köppen-Geiger raster ---
koppen_zip = os.path.join(cache_dir, 'Beck_KG_V1.zip')
koppen_tif = os.path.join(cache_dir, 'Beck_KG_V1_present_0p083.tif')
if not os.path.exists(koppen_zip):
    print('Downloading Beck et al. (2018) Köppen-Geiger data from figshare...')
    try:
        urllib.request.urlretrieve(
            'https://ndownloader.figshare.com/files/12407516', koppen_zip)
        print(f'  Saved: {koppen_zip}')
    except Exception as _dl_err:
        print(f'  [WARN] Koppen download failed: {_dl_err}')
        print('  Check network connection. Figure 2 may be incomplete.')
if not os.path.exists(koppen_tif):
    with zipfile.ZipFile(koppen_zip) as zf:
        zf.extract('Beck_KG_V1_present_0p083.tif', cache_dir)
    print(f'  Extracted: {koppen_tif}')

print('Reading Köppen-Geiger raster (0.083° ≈ 10 km)...')
if os.path.exists(koppen_tif):
    # Use tifffile for palette-mode TIFF (PIL's np.array conversion fails on some systems).
    try:
        import tifffile
        kg_global = tifffile.imread(koppen_tif)
    except ImportError:
        # Fallback to PIL with explicit conversion
        _img = Image.open(koppen_tif)
        kg_global = np.frombuffer(_img.tobytes(), dtype=np.uint8).reshape(_img.size[1], _img.size[0])
    print(f'  Shape: {kg_global.shape}, classes: {len(np.unique(kg_global))}')
else:
    print('[WARN] Koppen raster not found — Figure 2a will be simplified')
    kg_global = None

# --- 2. US state boundaries (Natural Earth 50m) ---
ne_states_path = os.path.join(cache_dir, 'ne_50m_admin_1_states_provinces.zip')
if not os.path.exists(ne_states_path):
    print('Downloading Natural Earth 50m state boundaries (timeout 30s)...')
    try:
        urllib.request.urlretrieve(
            'https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip',
            ne_states_path)
    except Exception as _ne_err:
        print(f'  [WARN] Natural Earth download failed: {_ne_err}')
states = gpd.read_file(ne_states_path)
us_states = states[states['admin'] == 'United States of America']
conus = us_states[~us_states['name'].isin(['Alaska', 'Hawaii'])]
georgia = conus[conus['name'] == 'Georgia']

# --- 3. GT campus data (OSMnx) — cached to GeoPackage ---
gpkg_path = os.path.join(cache_dir, 'gt_campus_osm.gpkg')

if os.path.exists(gpkg_path):
    print('Loading cached OSMnx data from', gpkg_path)
    gt_campus = gpd.read_file(gpkg_path, layer='campus')
    campus_roads_gdf = gpd.read_file(gpkg_path, layer='roads')
    campus_buildings = gpd.read_file(gpkg_path, layer='buildings')
    campus_center = gt_campus.geometry.iloc[0].centroid
else:
    print('Fetching Georgia Tech campus boundary...')
    gt_campus = ox.geocode_to_gdf('Georgia Institute of Technology, Atlanta, GA')
    campus_center = gt_campus.geometry.iloc[0].centroid
    fetch_lat, fetch_lon = campus_center.y, campus_center.x

    print('Fetching campus-area road network...')
    campus_roads = ox.graph_from_point((fetch_lat, fetch_lon), dist=1600, network_type='drive')
    campus_roads_gdf = ox.graph_to_gdfs(campus_roads, nodes=False)

    print('Fetching campus-area buildings...')
    campus_buildings = ox.features_from_point(
        (fetch_lat, fetch_lon), tags={'building': True}, dist=1600)
    campus_buildings = campus_buildings[
        campus_buildings.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

    # Cache for future runs
    gt_campus.to_file(gpkg_path, layer='campus', driver='GPKG')
    campus_roads_gdf.to_file(gpkg_path, layer='roads', driver='GPKG')
    campus_buildings.to_file(gpkg_path, layer='buildings', driver='GPKG')
    print(f'  Cached OSMnx data to {gpkg_path}')

# --- 4. Demuzere et al. (2022) LCZ raster (campus, ~100 m) for panel (d) ---
import json as _json
lcz_arr  = np.load(os.path.join(cache_dir, 'lcz_campus.npy'))
lcz_meta = _json.load(open(os.path.join(cache_dir, 'lcz_campus_meta.json')))
print(f'  LCZ raster: {lcz_arr.shape}, classes {sorted(int(x) for x in np.unique(lcz_arr))}')

print('All data loaded.')

# ════════════════════════════════════════════════════════════════════
# KOPPEN PROCESSING
# ════════════════════════════════════════════════════════════════════

# Beck et al. (2018) legend: class code → (RGB, short label, description)
koppen_info = {
    0:  (None,          'Ocean',  ''),
    1:  ([0,0,255],     'Af',     'Tropical rainforest'),
    2:  ([0,120,255],   'Am',     'Tropical monsoon'),
    3:  ([70,170,250],  'Aw',     'Tropical savanna'),
    4:  ([255,0,0],     'BWh',    'Hot desert'),
    5:  ([255,150,150], 'BWk',    'Cold desert'),
    6:  ([245,165,0],   'BSh',    'Hot semi-arid'),
    7:  ([255,220,100], 'BSk',    'Cold semi-arid'),
    8:  ([255,255,0],   'Csa',    'Hot-summer Mediterranean'),
    9:  ([200,200,0],   'Csb',    'Warm-summer Mediterranean'),
    10: ([150,150,0],   'Csc',    'Cold-summer Mediterranean'),
    11: ([150,255,150], 'Cwa',    'Humid subtropical (dry winter)'),
    12: ([100,200,100], 'Cwb',    'Subtropical highland'),
    13: ([50,150,50],   'Cwc',    'Cold subtropical highland'),
    14: ([200,255,80],  'Cfa',    'Humid subtropical'),
    15: ([100,255,80],  'Cfb',    'Oceanic'),
    16: ([50,200,0],    'Cfc',    'Subpolar oceanic'),
    17: ([255,0,255],   'Dsa',    'Mediterranean continental'),
    18: ([200,0,200],   'Dsb',    'Warm-summer Med. continental'),
    19: ([150,50,150],  'Dsc',    'Dry-summer subarctic'),
    20: ([150,100,150], 'Dsd',    'Dry-summer extreme continental'),
    21: ([170,175,255], 'Dwa',    'Monsoon humid continental'),
    22: ([90,120,220],  'Dwb',    'Warm-summer monsoon continental'),
    23: ([75,80,180],   'Dwc',    'Monsoon subarctic'),
    24: ([50,0,135],    'Dwd',    'Monsoon extreme continental'),
    25: ([0,255,255],   'Dfa',    'Hot-summer humid continental'),
    26: ([55,200,255],  'Dfb',    'Warm-summer humid continental'),
    27: ([0,125,125],   'Dfc',    'Subarctic'),
    28: ([0,70,95],     'Dfd',    'Extreme subarctic'),
    29: ([178,178,178], 'ET',     'Tundra'),
    30: ([102,102,102], 'EF',     'Ice cap'),
}

# Build color lookup table (class code → RGBA)
color_lut = np.full((31, 4), 255, dtype=np.uint8)
for code, (rgb, _, _) in koppen_info.items():
    if rgb is not None:
        color_lut[code] = [rgb[0], rgb[1], rgb[2], 255]

# Crop to CONUS
lon_min, lon_max = -128, -65
lat_min, lat_max = 24, 50
res = 360.0 / kg_global.shape[1]
col_s = int((lon_min + 180) / res)
col_e = int((lon_max + 180) / res)
row_s = int((90 - lat_max) / res)
row_e = int((90 - lat_min) / res)
kg_conus = kg_global[row_s:row_e, col_s:col_e]
rgba_conus = color_lut[kg_conus]

# ════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════
gt_bounds = gt_campus.total_bounds  # [minx, miny, maxx, maxy]

import matplotlib.gridspec as _mgs
fig = plt.figure(figsize=(FULL_W, 6.0799))  # v11.2 (2026-10-04): v11.1 rows kept to the 1e-5 in; only the white under b/c trimmed
_gs = _mgs.GridSpec(4, 2, figure=fig,   # rows in inches (hspace folded into the a-to-b/c band, which is unchanged)
    height_ratios=[1.83641, 0.65531, 2.53231, 0.70], hspace=0.0, wspace=0.05,
    left=0.050, right=0.995, top=0.941471, bottom=0.0,
    width_ratios=[0.50, 0.50])
ax_a = fig.add_subplot(_gs[0, 0])      # Koppen locator, top left, width-matched to b
ax_b = fig.add_subplot(_gs[2, 0])      # land-cover map (row 1 = seam spacer)
ax_d = fig.add_subplot(_gs[2, 1])      # LCZ map, same size/scale twin
ax_lc = fig.add_subplot(_gs[3, 0]); ax_lc.axis('off')      # cover legend UNDER b
ax_legd = fig.add_subplot(_gs[3, 1])   # LCZ legend UNDER c

for ax in [ax_a, ax_b, ax_d]:
    ax.grid(False)
    ax.set_facecolor('white')
ax_legd.axis('off')

# ════════════════════════════════════════════
# Panel (a): CONUS + Köppen climate zones
# ════════════════════════════════════════════

# Köppen raster background
ax_a.imshow(rgba_conus, extent=[lon_min, lon_max, lat_min, lat_max],
            aspect=1.3, interpolation='nearest', zorder=1)

# State boundaries
conus.boundary.plot(ax=ax_a, color='0.35', lw=0.3, zorder=2)

# Highlight Georgia
georgia.boundary.plot(ax=ax_a, color='k', lw=1.0, zorder=3)

# Atlanta marker (single point on CONUS map)
import matplotlib.patheffects as _pea
ax_a.plot(ATLANTA_LON, ATLANTA_LAT, marker='*', color='k', markersize=8.5,
          markeredgecolor='white', markeredgewidth=0.7, zorder=5)
_ann_atl = ax_a.annotate('Atlanta', xy=(ATLANTA_LON, ATLANTA_LAT),
              xytext=(ATLANTA_LON + 5.5, ATLANTA_LAT - 2.5),
              fontsize=6, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='0.25', lw=0.5,
                              shrinkA=1, shrinkB=3),
              zorder=6)
_ann_atl.arrow_patch.set_path_effects([_pea.withStroke(linewidth=1.4, foreground='white')])

# Cfa label near Atlanta
ax_a.text(ATLANTA_LON + 7, ATLANTA_LAT - 0.7, 'Cfa',
          fontsize=6, fontstyle='italic', color='0.15', zorder=6,
          path_effects=[_pea.withStroke(linewidth=1.6, foreground='white')])

# Zoom box around Georgia/Atlanta area for connector
zoom_pad = 2.0
ga_bounds = georgia.total_bounds
rect_left   = ga_bounds[0] - zoom_pad * 0.3
rect_bottom = ga_bounds[1] - zoom_pad * 0.3
rect_w      = ga_bounds[2] - ga_bounds[0] + zoom_pad * 0.6
rect_h      = ga_bounds[3] - ga_bounds[1] + zoom_pad * 0.6
rect_right  = rect_left + rect_w
rect_top    = rect_bottom + rect_h

rect_a = plt.Rectangle((rect_left, rect_bottom), rect_w, rect_h,
                        fill=False, edgecolor='0.3', lw=0.7, ls='--', zorder=4)
ax_a.add_patch(rect_a)

# Axes formatting
ax_a.set_xlim(lon_min, lon_max)
ax_a.set_ylim(lat_min, lat_max)
inline_panel_title(ax_a, 'a', 'K\u00f6ppen\u2013Geiger climate classification')   # project-standard title helper (2026-09-15)
from matplotlib.ticker import FuncFormatter
ax_a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.0f}'))
ax_a.set_ylabel('Latitude (°N)', fontsize=8)
ax_a.set_xlabel('Longitude (°W)', fontsize=8, labelpad=1.0)
ax_a.tick_params(labelsize=7)

# Köppen legend (compact, 6 major CONUS zones)
legend_items = [
    (14, 'Cfa \u2013 Humid subtropical'),
    (25, 'Dfa \u2013 Hot-summer continental'),
    (26, 'Dfb \u2013 Warm-summer continental'),
    (7,  'BSk \u2013 Cold semi-arid'),
    (4,  'BWh \u2013 Hot desert'),
    (8,  'Csa \u2013 Mediterranean'),
]
legend_patches = [Patch(facecolor=np.array(koppen_info[c][0]) / 255, label=lbl,
                        edgecolor='0.4', lw=0.3)
                  for c, lbl in legend_items]
ax_a.legend(handles=legend_patches, fontsize=6.5,
            loc='lower left', bbox_to_anchor=(1.042, 0.065),
            ncol=1, framealpha=0.95, facecolor="white", edgecolor='#cccccc',
            handlelength=1.1, handleheight=0.8, labelspacing=0.45, borderpad=0.5)
ax_a.get_legend().get_frame().set_linewidth(0.4)   # figure-wide legend frame: #cccccc, 0.4 pt (2026-09-17)

# ════════════════════════════════════════════
# Panel (b): Georgia Tech campus — ALL 10 STATIONS (5 land cover types)
# ════════════════════════════════════════════
view_pad = 0.005

# Campus boundary
ax_b.set_facecolor('#f5f5f0')   # Fig 10a ground tone
from shapely.geometry import Polygon as _Poly, MultiPolygon as _MPoly
_cg0 = gt_campus.geometry.iloc[0]
_parts = list(_cg0.geoms) if _cg0.geom_type == 'MultiPolygon' else [_cg0]
_amax = max(_p.area for _p in _parts)
_outer = _MPoly([_Poly(_p.exterior) for _p in _parts if _p.area >= 0.01 * _amax])   # outer rings only, tiny station-less parcels dropped
gpd.GeoSeries([_outer]).plot(ax=ax_b, facecolor='none', edgecolor='0.45', lw=0.7, ls='--', zorder=3.6)
ax_b.collections[-1].set_linestyle((0, (5, 2.2)))

# Buildings
campus_buildings.plot(ax=ax_b, facecolor='#e6e6e6', edgecolor='#c2c2c2', lw=0.25,
                      alpha=0.9, zorder=2)

# Roads
campus_roads_gdf.plot(ax=ax_b, color='#d8d8d8', lw=0.3, alpha=0.7, zorder=2)
try:
    _mw = campus_roads_gdf[campus_roads_gdf['highway'].astype(str).str.contains('motorway')]
    if len(_mw):
        _mw.plot(ax=ax_b, color='#c4c4c4', lw=1.4, alpha=0.9, zorder=2.1)
        print('   motorway emphasis: %d segments' % len(_mw))
except Exception as _e:
    print('   motorway emphasis skipped:', _e)

import matplotlib.patheffects as _pe   # needed by the marker casing below
# Plot all 10 stations with setting-based markers
plotted_settings = set()
for sid, info in STATIONS.items():
    sty = SETTING_STYLES[info['setting']]
    label = sty['label'] if info['setting'] not in plotted_settings else None
    plotted_settings.add(info['setting'])
    ax_b.plot(info['lon'], info['lat'],
              marker=sty['marker'], color=sty['color'], markersize=4,
              markeredgecolor='k', markeredgewidth=0.5, zorder=5, label=label,
              path_effects=[_pe.withStroke(linewidth=1.4, foreground='white')])
    _r_lat = 100.0 / 111320.0
    _r_lon = 100.0 / (111320.0 * np.cos(np.radians(info['lat'])))
    ax_b.add_patch(Ellipse((info['lon'], info['lat']), 2 * _r_lon, 2 * _r_lat,
                   facecolor='none', edgecolor='0.25', lw=1.1,
                   ls=(0, (2.5, 2.0)), zorder=4))

# Annotate each station with ID and short name
import matplotlib.patheffects as _pe
# Fig-10a-style labels: two-line block centered directly above/below each marker, no leaders
ann_side = {
    '10635576': ('above', '10th St\nBridge'),
    '10635580': ('above', '5th St\nBridge'),
    '10635581': ('right', 'Canopied\nWalkway'),
    '10635582': ('above', 'Hemphill\nParking'),
    '10635585': ('right', 'Baker\nBuilding'),
    '10635587': ('below', 'Curran\nParking'),
    '10635588': ('above', 'Constructed\nWetlands'),
    '10635590': ('below', 'Grass\nWalkway'),
    '10635593': ('below', 'Tech\nSquare'),
    '10635595': ('below', 'Brittain\nDrive'),
}
ann_nudge = {'10635585': 0.0009}     # Baker: clear its ring AND the Wetlands ring arc
ann_nudge_y = {'10635585': -0.0005, '10635593': -0.0003}
_BSIDE = {'10635582': 'right', '10635587': 'right', '10635581': 'right',
          '10635590': 'right', '10635585': 'right', '10635588': 'right',
          '10635576': 'right', '10635593': 'right', '10635595': 'below',
          '10635580': 'below'}   # same placement set as c's LCZ tags (author 2026-09-05)
_BGAP_LON, _BGAP_LAT = 0.000278, 0.000231   # 3 pt beyond the 100 m ring
for sid, info in STATIONS.items():
    _lbl = ann_side[sid][1]
    _rlat = 100.0 / 111320.0
    _rlon = 100.0 / (111320.0 * np.cos(np.radians(info['lat'])))
    side = _BSIDE[sid]
    if side == 'right':
        dx, dy = _rlon + _BGAP_LON, {'10635585': -0.0003}.get(sid, 0.0)
        ha, va = 'left', 'center'
    else:
        dx, dy = 0.0, -(_rlat + _BGAP_LAT)
        ha, va = 'center', 'top'
    ax_b.text(info['lon'] + dx, info['lat'] + dy, _lbl,
              ha=ha, va=va, fontsize=6, fontweight='bold',
              linespacing=0.95, zorder=6,
              path_effects=[_pe.withStroke(linewidth=1.8, foreground='white')])

# Station legend moved to panel (c) table

# Campus label inside boundary

# North arrow — upper-right corner
ax_b.annotate('N', xy=(0.965, 0.955), xytext=(0.965, 0.862), xycoords='axes fraction',
              ha='center', va='center', fontsize=8, fontweight='bold', zorder=9,
              arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black'))   # Fig 10a furniture

# 500 m scale bar inside panel, lower-left
lat_mid = (gt_bounds[1] + gt_bounds[3]) / 2
m_per_deg_lon = 111320 * np.cos(np.radians(lat_mid))
bar_deg = 200.0 / m_per_deg_lon
_pad = 0.0008  # tight crop around the campus boundary (author 2026-09-04)
_xlo_shared = max(gt_bounds[0] - _pad, lcz_meta['lon_min'] + 0.0002)
_xhi_shared = min(gt_bounds[2] + _pad, lcz_meta['lon_max'] - 0.0002)
_ylo_shared = max(gt_bounds[1] - _pad, lcz_meta['lat_min'] + 0.0002)
_yhi_shared = min(gt_bounds[3] + _pad, lcz_meta['lat_max'] - 0.0002)
bar_x = _xhi_shared - 0.0007 - bar_deg   # scale bar bottom-right (author 2026-09-05)
bar_y = _ylo_shared + 0.0010
# figure-ground (Fig 10a): wash the off-campus area toward white
from shapely.geometry import box as _shbox
_cg = gt_campus.geometry.iloc[0]
gpd.GeoSeries([_shbox(_xlo_shared - 0.001, _ylo_shared - 0.001,
                      _xhi_shared + 0.001, _yhi_shared + 0.001).difference(_outer)]).plot(
    ax=ax_b, facecolor='white', edgecolor='none', alpha=0.55, zorder=3)
ax_b.text(_xlo_shared + 0.0007, _ylo_shared + 0.0008,
          'Georgia Institute of Technology',
          ha='left', va='bottom', fontsize=6, fontstyle='italic',
          color='0.45', zorder=3.5,
          path_effects=[_pe.withStroke(linewidth=1.5, foreground='white')])
ax_b.plot([bar_x, bar_x + bar_deg], [bar_y, bar_y], color='black', lw=1.4, zorder=9,
          solid_capstyle='butt', path_effects=[_pe.withStroke(linewidth=2.8, foreground='white')])
ax_b.text(bar_x + bar_deg / 2, bar_y + 0.0004, '200 m', ha='center', va='bottom',
          fontsize=FS_DENSE_SM, zorder=9,
          path_effects=[_pe.withStroke(linewidth=2.2, foreground='white')])   # Fig 10a furniture

# OSM attribution
# Set view
ax_b.set_xlim(_xlo_shared, _xhi_shared)
ax_b.set_ylim(_ylo_shared, _yhi_shared)
ax_b.set_aspect(1.0 / np.cos(np.radians((gt_bounds[1] + gt_bounds[3]) / 2)))
_tb_ = inline_panel_title(ax_b, 'b', 'Stations and local land cover')
_tb_.set_path_effects([_pe.withStroke(linewidth=3.0, foreground='white')])
ax_b.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.3f}'))
ax_b.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ── Zoom connector lines: zoom-box corners → panel (b) axes corners ──
con_b = ConnectionPatch(
    xyA=(rect_left, rect_bottom), coordsA=ax_a.transData,
    xyB=(0, 1), coordsB=ax_b.transAxes,
    color='0.4', lw=0.7, ls='--', zorder=5)
con_c = ConnectionPatch(
    xyA=(rect_right, rect_bottom), coordsA=ax_a.transData,
    xyB=(1, 1), coordsB=ax_d.transAxes,
    color='0.4', lw=0.7, ls='--', zorder=5)
con_b.set_zorder(0.5)   # ABOVE panel a (fan visible from the box) but BELOW b/c (titles stay clean)
con_c.set_zorder(0.5)
for _axz in (ax_b, ax_d, ax_lc, ax_legd):
    _axz.set_zorder(1)
fig.add_artist(con_b)
fig.add_artist(con_c)

# ════════════════════════════════════════════
# Panel (c): Station characteristics table
# ════════════════════════════════════════════
import matplotlib.colors as _mcolors

# (panel-c station table removed 2026-09-04 -> supplement Table S14)



# ════════════════════════════════════════════
# Panel (d): Local Climate Zone map (Demuzere et al., 2022)
# ════════════════════════════════════════════
import matplotlib.patheffects as _pe2
_LCZ_COL  = {1:'#8c0000', 2:'#d10000', 3:'#ff0000', 4:'#bf4d00', 5:'#ff6600', 6:'#ff9955', 8:'#bcbcbc'}
_LCZ_NAME = {1:'Compact high-rise', 2:'Compact mid-rise', 3:'Compact low-rise', 4:'Open high-rise',
             5:'Open mid-rise', 6:'Open low-rise', 8:'Large low-rise'}
_le = lcz_meta
_rgba = np.zeros((lcz_arr.shape[0], lcz_arr.shape[1], 4))
for _v, _hex in _LCZ_COL.items():
    _mask = lcz_arr == _v
    _rr, _gg, _bb = _mcolors.to_rgb(_hex); _rgba[_mask] = [_rr, _gg, _bb, 1.0]
ax_d.imshow(_rgba, extent=[_le['lon_min'], _le['lon_max'], _le['lat_min'], _le['lat_max']],
            origin='upper', interpolation='nearest', zorder=1, aspect='auto')
campus_buildings.plot(ax=ax_d, facecolor='none', edgecolor='black', lw=0.3, alpha=0.30, zorder=2)
# (campus border removed from c, author 2026-09-05: b carries the campus delineation)
for _sid, _info in STATIONS.items():
    _sty = SETTING_STYLES[_info['setting']]
    ax_d.plot(_info['lon'], _info['lat'], marker=_sty['marker'], color=_sty['color'],
              markersize=4, markeredgecolor='k', markeredgewidth=0.5, zorder=5,
              path_effects=[_pe2.withStroke(linewidth=1.4, foreground='white')])
    _r_lat2 = 100.0 / 111320.0
    _r_lon2 = 100.0 / (111320.0 * np.cos(np.radians(_info['lat'])))
    _ring = Ellipse((_info['lon'], _info['lat']), 2 * _r_lon2, 2 * _r_lat2,
                    facecolor='none', edgecolor='0.25', lw=1.1, ls=(0, (2.5, 2.0)), zorder=4)
    _ring.set_path_effects([_pe2.withStroke(linewidth=2.3, foreground='white')])
    ax_d.add_patch(_ring)
    # per-station LCZ tag, mirroring b's label side (author 2026-09-05)
    _LCZ_OF = {'10635576': 8, '10635580': 8, '10635581': 5, '10635582': 5, '10635585': 5,
               '10635587': 8, '10635588': 5, '10635590': 8, '10635593': 2, '10635595': 5}
    # placement priority right > bottom > top (author 2026-09-05, per Fig 10a convention),
    # anchored 2 pt beyond the 100 m ring edge
    _GAP_LON, _GAP_LAT = 0.000278, 0.000231   # 3 pt at this panel scale
    _TAGPOS = {'10635582': 'right', '10635587': 'right', '10635581': 'right',
               '10635590': 'right', '10635585': 'right', '10635588': 'right',
               '10635576': 'right', '10635593': 'right', '10635595': 'below',
               '10635580': 'below'}   # 5th St: right side sits inside Tech Sq's ring
    _side = _TAGPOS[_sid]
    if _side == 'right':
        _dx2, _dy2 = _r_lon2 + _GAP_LON, {'10635585': -0.0003}.get(_sid, 0.0)
        _ha2, _va2 = 'left', 'center'
    else:
        _dx2, _dy2 = 0.0, -(_r_lat2 + _GAP_LAT)
        _ha2, _va2 = 'center', 'top'
    ax_d.text(_info['lon'] + _dx2, _info['lat'] + _dy2, 'LCZ %d' % _LCZ_OF[_sid],
              ha=_ha2, va=_va2, fontsize=6, fontweight='bold', zorder=6,
              path_effects=[_pe2.withStroke(linewidth=1.8, foreground='white')])
ax_d.set_xlim(_xlo_shared, _xhi_shared); ax_d.set_ylim(_ylo_shared, _yhi_shared)
ax_d.set_aspect(1.0 / np.cos(np.radians((_le['lat_min'] + _le['lat_max']) / 2)))
_tc_ = inline_panel_title(ax_d, 'c', 'Local Climate Zones')
_tc_.set_path_effects([_pe2.withStroke(linewidth=3.0, foreground='white')])
ax_d.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):.3f}'))
# axis furniture removed (author 2026-09-04): scale bar + north arrow carry the georeference
ax_d.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# (Midtown label removed by author 2026-09-05)
# (Campus area label removed by author 2026-09-05)
_nad = ax_d.annotate('N', xy=(0.965, 0.955), xytext=(0.965, 0.862), xycoords='axes fraction',
              ha='center', va='center', fontsize=8, fontweight='bold', zorder=9,
              path_effects=[_pe2.withStroke(linewidth=1.8, foreground='white')],
              arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black'))   # Fig 10a furniture
_nad.arrow_patch.set_path_effects([_pe2.withStroke(linewidth=3.0, foreground='white')])
ax_d.plot([bar_x, bar_x + bar_deg], [bar_y, bar_y], color='black', lw=1.4, zorder=9,
          solid_capstyle='butt', path_effects=[_pe2.withStroke(linewidth=2.8, foreground='white')])
ax_d.text(bar_x + bar_deg / 2, bar_y + 0.0004, '200 m', ha='center', va='bottom',
          fontsize=FS_DENSE_SM, zorder=9,
          path_effects=[_pe2.withStroke(linewidth=2.2, foreground='white')])   # Fig 10a furniture

# LCZ legend + descriptive note (ax_legd)
_present = sorted(int(x) for x in np.unique(lcz_arr) if int(x) in _LCZ_COL)
_lcz_patches = [Patch(facecolor=_LCZ_COL[v], edgecolor='0.4', lw=0.3,
                      label=f'LCZ {v} – {_LCZ_NAME[v]}') for v in _present]
_legd = ax_legd.legend(handles=_lcz_patches, fontsize=6.5, loc='center',
                       bbox_to_anchor=(0.5, 0.48), ncol=2, framealpha=0.95,
                       facecolor='white', edgecolor='#cccccc',
                       title='Local Climate Zone (100 m)', handlelength=1.0, handleheight=0.85,
                       labelspacing=0.40, borderpad=0.5, columnspacing=0.8, handletextpad=0.4)
_legd.get_title().set_fontweight('bold'); _legd.get_title().set_fontsize(8); _legd.get_frame().set_linewidth(0.4)
# (LCZ note removed by author 2026-09-04 -> content goes to the caption)

_cov_handles = [plt.Line2D([], [], marker=SETTING_STYLES[_s]['marker'], color=SETTING_STYLES[_s]['color'],
                linestyle='None', markersize=7, markeredgecolor='k', markeredgewidth=0.7,
                label=SETTING_STYLES[_s]['label']) for _s in SETTING_STYLES]
from matplotlib.legend_handler import HandlerPatch
class _HRing(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        _p = Ellipse((width / 2 - xdescent, height / 2 - ydescent), height * 1.15, height * 1.15,
                     facecolor='none', edgecolor='0.25', lw=1.1, ls=(0, (2.5, 2.0)))
        _p.set_transform(trans)
        return [_p]
_ring_key = Ellipse((0, 0), 1, 1, facecolor='none', edgecolor='0.25', lw=1.1,
                    ls=(0, (2.5, 2.0)), label='100 m radius')
_cov_handles.append(_ring_key)
_legc = ax_lc.legend(handles=_cov_handles, fontsize=6.5, loc='center',
                     handler_map={Ellipse: _HRing()},
                     bbox_to_anchor=(0.5, 0.48), ncol=3, framealpha=0.95,
                     facecolor='white', edgecolor='#cccccc',
                     title='Station land cover (b, c)', handlelength=1.0,
                     labelspacing=0.55, borderpad=0.5, columnspacing=1.2, handletextpad=0.5)
_legc.get_title().set_fontweight('bold'); _legc.get_title().set_fontsize(8); _legc.get_frame().set_linewidth(0.4)

# headroom sentinel: the fullwidth save crops flush to the topmost ink; this invisible
# dot 0.03 in above the title keeps glyph tops off the canvas edge (render check 2026-09-05)
fig.canvas.draw()
_tb = fig.get_tightbbox(fig.canvas.get_renderer())
fig.text(0.5, (_tb.y1 + 0.03) / fig.get_size_inches()[1], '.', color='white', fontsize=2, va='center')
savefig(plt.gcf(), 'fig02_station_map', bbox='fullwidth')
plt.close()

print('Figure 2 saved: fig02_station_map.png + fig02_station_map.pdf')
print()
print('--- CAPTION (copy to manuscript) ---')
print('See manuscript for the Figure 2 caption (rewritten 2026-09-04 with the v7A redesign).')
