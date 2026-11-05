"""
Shared figure configuration for Urban Climate publication.
Import this in generate_figures.ipynb and standalone _build_*.py scripts.

Usage:
    from fig_config import *
    # or selectively:
    from fig_config import FULL_W, DPI, MODEL_COLORS, apply_rcparams
"""
import matplotlib.pyplot as plt

# =====================================================================
# Journal: Elsevier / Urban Climate (Q1)
# =====================================================================
DPI = 1000   # 13 Oct 2026: Urban Climate "Bitmapped line drawings ... Minimum resolution 1000 dpi" (full page 7480 px); was 600
FULL_W = 7.48    # double-column width (190 mm)
MEDIUM_W = 5.51  # 1.5-column width (140 mm)
SINGLE_W = 3.54  # single-column width (90 mm)
MAX_H = 9.45     # maximum figure height (240 mm)

# =====================================================================
# Model colors (colorblind-safe, consistent across all figures)
# 4 candidate models + 2 benchmarks
# =====================================================================
MODEL_COLORS = {
    'PatchTST':       '#0072B2',  # blue
    'iTransformer':   '#785EF0',  # purple
    'LSTM':           '#E69F00',  # amber
    'Attention-LSTM': '#D55E00',  # vermillion
    'DLinear':        '#56B4E9',  # light blue (linear benchmark)
    'Persistence':    '#999999',  # gray (naive benchmark)
}
MODEL_SHORT = {
    'PatchTST': 'PT', 'iTransformer': 'iT',
    'LSTM': 'LS', 'Attention-LSTM': 'AL',
}
CANDIDATES = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']

# =====================================================================
# Land cover / LCZ colors
# =====================================================================
LC_COLORS = {
    'Tree canopy': '#2ca02c', 'Open green': '#7fbf7f',
    'Impervious': '#7f7f7f', 'Built': '#d62728', 'Exposed': '#ff7f0e',
}
LCZ_COLORS = {2: '#d10000', 5: '#4A90D9', 8: '#E6A817'}
VEG_COLORS = {'Vegetated': '#2ca02c', 'Non-vegetated': '#8B4513'}

# Three-regime ladder (shared across Fig 3, Fig 6, Fig 8+)
# Light fills for axvspan backgrounds; saturated for label text.
REGIME_BG_COLORS = {1: '#e8f4fd', 2: '#fff8e1', 3: '#fce4ec'}
REGIME_COLOR     = {1: '#4A90D9', 2: '#E6A817', 3: '#D35F5F'}
REGIME_NAMES     = {1: 'Thermal inertia', 2: 'Diurnal decoupling', 3: 'Full diurnal cycle'}

# =====================================================================
# Station coordinates (SINGLE SOURCE OF TRUTH for all station metadata)
# Every other station-level structure (ORDERED_STATIONS_LCZ, SHORT_NAMES)
# is derived from this dict.
# =====================================================================
STATION_COORDS = {
    10635576: {'name': '10th St Bridge',       'lat': 33.781539, 'lon': -84.391662, 'lc': 'Exposed',     'lcz': 8},
    10635580: {'name': '5th St Bridge',        'lat': 33.776853, 'lon': -84.390820, 'lc': 'Exposed',     'lcz': 8},
    10635581: {'name': 'Canopied Walkway',     'lat': 33.777870, 'lon': -84.400738, 'lc': 'Tree canopy', 'lcz': 5},
    10635582: {'name': 'Hemphill Parking',     'lat': 33.781009, 'lon': -84.405840, 'lc': 'Impervious',  'lcz': 5},
    10635585: {'name': 'Baker Building',       'lat': 33.780249, 'lon': -84.399261, 'lc': 'Built',       'lcz': 5},
    10635587: {'name': 'Curran Parking Deck',  'lat': 33.778978, 'lon': -84.405711, 'lc': 'Impervious',  'lcz': 8},
    10635588: {'name': 'Constructed Wetlands', 'lat': 33.781267, 'lon': -84.398088, 'lc': 'Open green',  'lcz': 5},
    10635590: {'name': 'Grass Walkway',        'lat': 33.775932, 'lon': -84.400763, 'lc': 'Open green',  'lcz': 8},
    10635593: {'name': 'Tech Square',          'lat': 33.776800, 'lon': -84.389269, 'lc': 'Built',       'lcz': 2},
    10635595: {'name': 'Brittain Drive',       'lat': 33.774751, 'lon': -84.393105, 'lc': 'Tree canopy', 'lcz': 5},
}

# =====================================================================
# Station display order for LCZ-rooted hierarchy (Figs 5, 9, 10)
# Non-vegetated first, then Vegetated; within each group, LC order
# (Built → Impervious → Exposed  /  Open green → Tree canopy).
# This list is the ONLY ordering decision; the rich tuples below are
# derived from STATION_COORDS.
# =====================================================================
_STATION_ORDER_IDS = [
    10635593, 10635585,              # Built:       Tech Sq (LCZ 2), Baker (LCZ 5)
    10635582, 10635587,              # Impervious:  Hemphill (5),    Curran (8)
    10635576, 10635580,              # Exposed:     10th (8),        5th (8)
    10635590, 10635588,              # Open green:  Grass Walk (8),  Wetlands (5)
    10635581, 10635595,              # Tree canopy: Canopied (5),    Brittain (5)
]

_VEGETATED_LC = {'Tree canopy', 'Open green'}


def _surface_type(lc):
    """Classify a land cover as Vegetated or Non-vegetated."""
    return 'Vegetated' if lc in _VEGETATED_LC else 'Non-vegetated'


# Derived: (station_id, name, land_cover, lcz, surface_type) tuples in
# display order. Previously hand-maintained; now auto-generated so a change
# to STATION_COORDS (e.g., LCZ correction) propagates automatically.
ORDERED_STATIONS_LCZ = [
    (sid,
     STATION_COORDS[sid]['name'],
     STATION_COORDS[sid]['lc'],
     STATION_COORDS[sid]['lcz'],
     _surface_type(STATION_COORDS[sid]['lc']))
    for sid in _STATION_ORDER_IDS
]

SHORT_NAMES = {
    'Constructed Wetlands': 'Constr. Wetlands',
    'Curran Parking Deck': 'Curran Parking',
}
LCZ_SHORT = {2: 'LCZ 2', 5: 'LCZ 5', 8: 'LCZ 8'}
VEG_ORDER = ['Non-vegetated', 'Vegetated']

# =====================================================================
# Horizons
# =====================================================================
HORIZONS = ['5min', '10min', '1h', '6h', '24h']
HZ_LABELS = {'5min': '5 min', '10min': '10 min', '1h': '1 h', '6h': '6 h', '24h': '24 h'}
REGIME_LABELS = {
    '5min': 'Thermal inertia', '10min': 'Thermal inertia', '1h': 'Thermal inertia',
    '6h': 'Diurnal decoupling', '24h': 'Full diurnal cycle',
}

# =====================================================================
# Dendrogram layout constants
# All figures use the same relative positions (scale x by width ratio)
# =====================================================================
DENDRO = {
    'VEG_X2': 2.5,    # surface cover label center
    'J0_X':   4.0,    # junction veg -> lcz
    'LCZ_X2': 6.0,    # LCZ label center
    'J1_X':   7.5,    # junction lcz -> lc
    'LC_X2':  10.5,   # LC label center
    'J2_X':   12.0,   # junction lc -> station
    'STA_X':  12.8,   # station name x
    'VEG_R':  3.5,    # surface cover box right edge
    'LCZ_R':  6.75,   # LCZ box right edge
    'LC_R':   11.6,   # LC box right edge
    'XLIM':   20,     # ax_tree xlim
}

# =====================================================================
# Feature process group colors (shared between fig06, fig07, fig08)
# =====================================================================
FEATURE_GROUP_COLORS = {
    'Thermal inertia':            '#d62728',  # warm red — heat/autocorrelation
    'Diurnal memory':             '#a0522d',  # sienna — still temperature, distinct shade
    'Diurnal cycle position':     '#f39c12',  # orange — time/sun cycle
    'Seasonal cycle':             '#bcbd22',  # olive — time, less prominent
    'Moisture / evapotranspiration': '#5f9ea0',  # muted teal — cool/water, not blue (colorbar)
}

# =====================================================================
# Default palette (colorblind-safe, 7 colors)
# =====================================================================
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']

# =====================================================================
# Font size constants (Elsevier minimum: 7pt normal, 6pt subscript)
# Use these in ax.text() calls that don't auto-inherit rcParams.
# =====================================================================
FS_TITLE = 9       # axes.titlesize — figure/panel titles
FS_LABEL = 8       # axes.labelsize — axis labels, horizon labels, root labels
FS_TICK = 7        # xtick/ytick.labelsize — tick labels, model names, annotations
FS_LEGEND = 7      # legend.fontsize — legend entries, regime labels
FS_DENSE = 6       # dense figure cells — sig% inside tiles (Elsevier 6pt floor)
FS_DENSE_SM = 6    # dense figure cells — n.s. text, % inside bars (Elsevier 6pt floor)

# =====================================================================
# Panel label formatting (Elsevier standard)
# STANDARD (adopted from Fig. 2, 2026-09-15): ONE string "a) Description",
# 9 pt bold, left-aligned at the panel top, via inline_panel_title().
# =====================================================================
PANEL_LABEL_SIZE = 9       # same as FS_TITLE
PANEL_LABEL_WEIGHT = 'bold'
PANEL_LABEL_LOC = 'left'   # matplotlib set_title loc parameter
PANEL_LABEL_PAD = 14       # padding (pts) between panel top edge and label
PANEL_SUBTITLE_SIZE = 8    # same as FS_LABEL
PANEL_SUBTITLE_LOC = 'center'
PANEL_SUBTITLE_OFFSET = 1.01  # y position in axes coords for subtitle text


# =====================================================================
# Dendrogram styling (shared between fig05 and fig08d)
# =====================================================================
# Line widths (3 hierarchy levels)
LW_ROOT = 1.0      # root trunk + root-to-veg connectors
LW_BRANCH = 0.8    # veg-to-lcz, lcz-to-lc connectors
LW_LEAF = 0.5      # lc-to-station connectors

# Box padding
PAD_VEG = 0.15     # surface cover boxes (Non-veg, Vegetated)
PAD_LCZ = 0.12     # LCZ boxes
PAD_LC = 0.10      # land cover boxes

# Connector colors (neutral grays, hierarchy-based)
COL_ROOT = '#555555'     # root trunk
COL_CONNECTOR = '#888888' # neutral connectors (where parent color not used)

# Box edge line widths
BLW_VEG = 0.8      # surface cover box edge
BLW_LCZ = 0.5      # LCZ box edge
BLW_LC = 0.4       # land cover box edge

# =====================================================================
# Panel spacing (for multi-panel figures with GridSpec)
# =====================================================================
HSPACE_PANELS = 0.25   # vertical gap between stacked panels (e.g. heatmap + bar)
WSPACE_PANELS = 0.12   # horizontal gap between side-by-side panels (e.g. horizons)
WSPACE_CBAR = 0.45     # horizontal gap when left panel has a colorbar (e.g. dot matrix + bar chart)

# =====================================================================
# Figure margins (subplots_adjust values)
# =====================================================================
# Elsevier does NOT require figure margins — figures should fill their area.
# Use near-zero margins; each figure manages spacing via hspace/wspace/pad.
# Individual figures can override if needed (e.g., dendrograms need left space).
MARGIN_TOP = 0.98
MARGIN_BOTTOM = 0.02
MARGIN_LEFT = 0.02
MARGIN_RIGHT = 0.98

# =====================================================================
# Debug: show figure margins and panel boundaries
# Set DEBUG_MARGINS = False to draw red dashed lines at figure margins
# =====================================================================
DEBUG_MARGINS = False

def draw_debug_margins(fig):
    """Draw red dashed lines at figure margins for layout debugging.
    Call after all panels are created, before savefig."""
    if not DEBUG_MARGINS:
        return
    import matplotlib.patches as mpatches
    # Draw figure margin rectangle
    margin_rect = mpatches.FancyBboxPatch(
        (MARGIN_LEFT, MARGIN_BOTTOM),
        MARGIN_RIGHT - MARGIN_LEFT,
        MARGIN_TOP - MARGIN_BOTTOM,
        boxstyle='square,pad=0', fill=False,
        edgecolor='red', linestyle='--', linewidth=0.8,
        transform=fig.transFigure, zorder=999)
    fig.patches.append(margin_rect)
    # Draw each axes boundary in blue
    for ax in fig.axes:
        pos = ax.get_position()
        ax_rect = mpatches.FancyBboxPatch(
            (pos.x0, pos.y0), pos.width, pos.height,
            boxstyle='square,pad=0', fill=False,
            edgecolor='blue', linestyle=':', linewidth=0.5,
            transform=fig.transFigure, zorder=998)
        fig.patches.append(ax_rect)
    # Draw horizontal gap lines (green) between vertically adjacent panels
    positions = sorted(set(
        [(round(ax.get_position().y0, 4), round(ax.get_position().y1, 4))
         for ax in fig.axes]), key=lambda x: x[0])
    for i in range(len(positions) - 1):
        gap_bottom = positions[i][1]
        gap_top = positions[i + 1][0]
        if gap_top - gap_bottom > 0.005:  # significant gap
            mid_y = (gap_top + gap_bottom) / 2
            fig.lines.append(plt.Line2D(
                [0, 1], [mid_y, mid_y],
                transform=fig.transFigure, color='green',
                linestyle='-.', linewidth=0.5, zorder=997))
            # Label gap size
            fig.text(0.01, mid_y, f'{(gap_top-gap_bottom)*100:.1f}%',
                     transform=fig.transFigure, fontsize=5,
                     color='green', va='center')


# =====================================================================
# OSM overlay colors (for spatial figures with building/road context)
# =====================================================================
OSM_BUILDING_FACE = '#a0a0a0'   # 50% darker than previous #d4d4d4
OSM_BUILDING_EDGE = '#777777'   # 50% darker than previous #aaaaaa
OSM_BUILDING_LW = 0.3
OSM_BUILDING_ALPHA = 0.7
OSM_ROAD_COLOR = '#666666'      # 50% darker than previous #999999
OSM_ROAD_LW = 0.5
OSM_ROAD_ALPHA = 0.6

# =====================================================================
# Panel title helpers — STANDARD for Urban Climate / Elsevier figures
# =====================================================================
def inline_panel_title(ax, letter, desc, color_desc='black', pad_pts=None,
                        fontsize_letter=None, fontsize_desc=None,
                        gs_cell=None):
    """Panel title in the PROJECT STANDARD form (adopted 2026-09-15 from
    Fig. 2): ONE string "a) Description", one size (PANEL_LABEL_SIZE, 9 pt),
    bold, flush with the panel's left edge, baseline-aligned exactly as
    matplotlib's set_title(loc='left') places it. Returns the Text artist
    (so callers can add path effects, as Fig. 2 does).

    REFERENCE FRAME:
    - Default: relative to AXIS top (ax.transAxes). Works when the axis
      fills its gridspec cell.
    - With gs_cell: relative to the gridspec CELL top (true "offset from
      panel"). Use when the axis is smaller than its cell (set_aspect /
      set_box_aspect / set_anchor) - title aligns with sibling cells in
      the same row regardless of axis size.

    Parameters
    ----------
    ax : matplotlib axes
    letter : str, bare panel letter, e.g. 'a' (renders as 'a) '). Pass ''
        to suppress the letter (continuation sub-panels under one label).
    desc : str, descriptive title text
    color_desc : str, title color (default black)
    pad_pts : float or None, vertical offset in POINTS above the reference
        top. None = rcParams['axes.titlepad'] (8 pt under apply_rcparams),
        i.e. identical to set_title. Use a larger value (e.g. 18) if the
        axes have top-mounted x-tick labels.
    fontsize_letter, fontsize_desc : optional size override; the larger of
        the two is used for the whole string (single-size standard).
    gs_cell : optional gridspec cell (see above).
    """
    import matplotlib as _mpl
    fs = max(fontsize_letter or 0, fontsize_desc or 0) or PANEL_LABEL_SIZE
    pad = float(_mpl.rcParams['axes.titlepad']) if pad_pts is None else pad_pts
    text = f'{letter}) {desc}' if letter else desc

    if gs_cell is not None:
        fig = ax.figure
        cell_bbox = gs_cell.get_position(fig)
        pad_fig_y = pad / 72.0 / fig.get_figheight()
        return fig.text(cell_bbox.x0, cell_bbox.y1 + pad_fig_y, text,
                        fontsize=fs, fontweight='bold', color=color_desc,
                        va='baseline', ha='left')
    from matplotlib.transforms import offset_copy
    trans = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=pad,
                        units='points')
    return ax.text(0.0, 1.0, text, transform=trans, fontsize=fs,
                   fontweight='bold', color=color_desc,
                   va='baseline', ha='left')


# =====================================================================
# Colorbar tick formatting (visual utility, not data-dependent)
# =====================================================================
def make_cbar_ticks(vmin, vmax, step=0.2):
    """Generate clean colorbar tick positions with key values (vmin, 0, vmax) bolded.
    Intermediate ticks at `step` intervals, clipped to [vmin, vmax].
    Returns (tick_positions, tick_labels, bold_indices)."""
    import numpy as _np
    # Key values always included
    key = {vmin, 0.0, vmax}
    # Add round intermediate ticks within range
    mid = set(_np.arange(step, vmax + step / 2, step))
    mid = {round(v, 2) for v in mid if v < vmax - step / 4}
    ticks = sorted(key | mid)
    labels = [f'{b:.2f}' if abs(b) < 0.1 else f'{b:.1f}' for b in ticks]
    bold_idx = [i for i, b in enumerate(ticks) if b in key]
    return ticks, labels, bold_idx


def apply_rcparams():
    """Apply publication-quality matplotlib defaults.

    Note: this sets ``savefig.bbox = 'tight'`` globally. The tight bbox will
    EXPAND the saved image beyond ``figsize`` if any element (legend, long
    caption text, annotation) extends outside the figure's axes area — and
    will SHRINK it if whitespace surrounds the axes. If a script must stick
    exactly to ``figsize`` (e.g. to match Elsevier double-column 7.48 in),
    call ``savefig(fig, ..., bbox='standard')`` instead of relying on the
    ``'tight'`` default. See ``savefig()`` docstring below.
    """
    plt.rcParams.update({
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'bold',
        'axes.titlepad': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.figsize': (FULL_W, 4),
        'axes.grid': False,
        'grid.alpha': 0.3,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.85',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
    })


def savefig(fig_or_plt, basename, dpi=DPI, pad=0.03, bbox='tight'):
    """Save figure as PNG + PDF with consistent settings.

    Parameters
    ----------
    fig_or_plt : Figure or plt module
    basename   : str, e.g. 'fig09' (saves fig09.png + fig09.pdf)
    dpi        : int, default DPI from config
    pad        : float, pad_inches for bbox_inches='tight'
    bbox       : str, 'tight' (default, prevents clipping) or 'standard'
                 (enforces exact figsize; needed when overflow elements
                 would otherwise expand the saved canvas).
    """
    kw = dict(dpi=dpi, facecolor='white')
    if bbox == 'tight':
        kw['bbox_inches'] = 'tight'
        kw['pad_inches'] = pad
        fig_or_plt.savefig(f'{basename}.png', **kw)
        kw['dpi'] = DPI
        fig_or_plt.savefig(f'{basename}.pdf', **kw)
    elif bbox == 'fullwidth':
        # Exact FULL_W width, TIGHT vertical crop. Captures any vertically
        # overflowing element (label/title sitting outside the axes) via the
        # tight bbox — content is never clipped — while forcing the horizontal
        # extent to exactly FULL_W so every figure shares one print scale.
        # Content narrower than FULL_W is centred with symmetric white margins.
        from matplotlib.transforms import Bbox
        fig = fig_or_plt if hasattr(fig_or_plt, 'get_tightbbox') else fig_or_plt.gcf()
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())   # inches
        cx = 0.5 * (tb.x0 + tb.x1)
        # `pad` inches of vertical breathing room (2026-09-16): with zero pad the tight bbox
        # cuts the outer half of any frame stroke (bottom legends of Figs 7/8) and lets
        # title ascenders touch the canvas edge. Width is still exactly FULL_W.
        nb = Bbox.from_extents(cx - FULL_W / 2.0, tb.y0 - pad, cx + FULL_W / 2.0, tb.y1 + pad)
        kw['bbox_inches'] = nb
        kw['pad_inches'] = 0
        fig.savefig(f'{basename}.png', **kw)
        kw['dpi'] = DPI
        fig.savefig(f'{basename}.pdf', **kw)
    else:
        # 'standard' — temporarily switch the global rcParam from 'tight'
        # (set by apply_rcparams) to 'standard' so figsize controls output.
        _saved_bbox = plt.rcParams['savefig.bbox']
        plt.rcParams['savefig.bbox'] = 'standard'
        try:
            fig_or_plt.savefig(f'{basename}.png', **kw)
            kw['dpi'] = DPI
            fig_or_plt.savefig(f'{basename}.pdf', **kw)
        finally:
            plt.rcParams['savefig.bbox'] = _saved_bbox
    print(f'Saved {basename}.png and {basename}.pdf')