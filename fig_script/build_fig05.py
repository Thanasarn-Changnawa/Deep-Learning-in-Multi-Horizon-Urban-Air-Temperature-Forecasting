"""Standalone builder for Fig 5: feature importance.
Synced from generate_figures.ipynb cell 12.

Panels A-E: Heatmap (features x horizons) per model -- shows EVERY cell of the data cube.
Panel F: Crossover summary (% temporal vs horizon, candidates vs DLinear).
The color gradient shift from left (lag-dominated) to right (temporal-dominated) IS the crossover.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from fig_config import (
    PANEL_LABEL_SIZE,
    FULL_W, DPI, MODEL_COLORS, HORIZONS, HZ_LABELS,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE, FS_DENSE_SM,
    HSPACE_PANELS, WSPACE_PANELS, FEATURE_GROUP_COLORS,
    MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT,
    REGIME_NAMES, REGIME_COLOR,
    make_cbar_ticks,
    apply_rcparams,
)
apply_rcparams()

# ── Data ──
imp = pd.read_csv('multihorizon_importance.csv')

HZ_LABELS_LIST = [HZ_LABELS[h] for h in HORIZONS]  # config dict -> ordered list
ALL_MODELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM', 'DLinear']
MODEL_DISPLAY = {'PatchTST': 'PatchTST', 'iTransformer': 'iTransformer',
                 'LSTM': 'LSTM', 'Attention-LSTM': 'Attn-LSTM', 'DLinear': 'DLinear'}

# Feature category colors (for y-axis label coloring)
CAT_COLORS = {
    'Temperature': '#1f77b4',
    'Humidity':    '#2ca02c',
    'Temporal':    '#ff7f0e',
}

# Feature display names
# All 30 features → human-readable display names
FEAT_DISPLAY = {
    # Temperature-derived (18)
    'Temp_rmean_12':  'Temp 1h mean',   'Temp_rmean_72':  'Temp 6h mean',
    'Temp_rmean_288': 'Temp 24h mean',
    'Temp_lag12':     'Temp lag-1h',     'Temp_lag24':     'Temp lag-2h',
    'Temp_lag72':     'Temp lag-6h',     'Temp_lag144':    'Temp lag-12h',
    'Temp_lag288':    'Temp lag-24h',
    'Temp_roc':       'Temp RoC',        'Temp_roc_12':    'Temp RoC-1h',
    'Temp_rstd_12':   'Temp 1h std',     'Temp_rstd_72':   'Temp 6h std',
    'Temp_rstd_288':  'Temp 24h std',
    'Temp_rmax_288':  'Temp 24h max',    'Temp_rmin_288':  'Temp 24h min',
    'Temp_dtr_288':   'Temp DTR-24h',
    'Temp':           'Temp (raw)',
    # Humidity-derived (10)
    'RH_rmean_12':    'RH 1h mean',      'RH_rmean_72':    'RH 6h mean',
    'RH_rmean_288':   'RH 24h mean',
    'RH_lag12':       'RH lag-1h',        'RH_lag24':       'RH lag-2h',
    'RH_lag72':       'RH lag-6h',        'RH_lag144':      'RH lag-12h',
    'RH_lag288':      'RH lag-24h',
    'RH_roc':         'RH RoC',
    'RH':             'RH (raw)',
    # Temporal encoding (4)
    'hour_sin':       'Hour (sin)',       'hour_cos':       'Hour (cos)',
    'doy_sin':        'DoY (sin)',        'doy_cos':        'DoY (cos)',
}

def categorize(f):
    if any(k in f for k in ['hour_', 'doy_', 'month_']):
        return 'Temporal'
    if f.startswith('RH') or f == 'RH':
        return 'Humidity'
    return 'Temperature'

def get_importance(model, horizon):
    sub = imp[(imp['Model'] == model) & (imp['Horizon'] == horizon)]
    return sub.groupby('Feature')['Importance'].mean()

SIG_THRESHOLD = 0.02  # practical significance: CI lower bound must exceed this (°C)

def is_significant(model, horizon, feature):
    """Practically significant: 95% CI lower bound > 0.02 °C (n=10 stations)."""
    sub = imp[(imp['Model'] == model) & (imp['Horizon'] == horizon)
              & (imp['Feature'] == feature)]
    vals = sub['Importance'].values
    if len(vals) < 3:
        return False
    mean_val = np.mean(vals)
    se = np.std(vals, ddof=1) / np.sqrt(len(vals))
    return mean_val > 0 and (mean_val - 1.96 * se) > SIG_THRESHOLD

# ── Build shared feature order ──
# Rank features by max importance across ALL models and horizons (most important at top)
all_feats_set = set()
for m in ALL_MODELS:
    for hz in HORIZONS:
        all_feats_set |= set(get_importance(m, hz).index)

# Max importance across all models × all horizons
global_max = {}
for f in all_feats_set:
    vals = []
    for m in ALL_MODELS:
        for hz in HORIZONS:
            s = get_importance(m, hz)
            vals.append(s.get(f, 0))
    global_max[f] = max(vals)

# Sort: group by 5 physical process groups, then by importance within group
# Order: Thermal inertia → Diurnal memory → Diurnal cycle → Seasonal → Moisture/ET
PROCESS_GROUPS = {
    'Thermal inertia\n(recent state)': {
        'features': {'Temp_rmean_12', 'Temp_lag12', 'Temp_lag24', 'Temp_rmean_72',
                      'Temp_lag72', 'Temp_lag144', 'Temp_roc_12', 'Temp_roc', 'Temp_rstd_12'},
        'order': 0, 'color': FEATURE_GROUP_COLORS['Thermal inertia'],
    },
    'Diurnal memory\n(24h cycle)': {
        'features': {'Temp_lag288', 'Temp_rmean_288', 'Temp_rmin_288', 'Temp_rmax_288',
                      'Temp_dtr_288', 'Temp_rstd_288', 'Temp_rstd_72'},
        'order': 1, 'color': FEATURE_GROUP_COLORS['Diurnal memory'],
    },
    'Diurnal cycle\nposition': {
        'features': {'hour_sin', 'hour_cos'},
        'order': 2, 'color': FEATURE_GROUP_COLORS['Diurnal cycle position'],
    },
    'Seasonal\ncycle': {
        'features': {'doy_sin', 'doy_cos'},
        'order': 3, 'color': FEATURE_GROUP_COLORS['Seasonal cycle'],
    },
    'Moisture /\nevapotranspiration': {
        'features': {'RH', 'RH_rmean_12', 'RH_rmean_72', 'RH_rmean_288',
                      'RH_lag12', 'RH_lag24', 'RH_lag72', 'RH_lag144', 'RH_lag288', 'RH_roc'},
        'order': 4, 'color': FEATURE_GROUP_COLORS['Moisture / evapotranspiration'],
    },
}

def get_process_group(f):
    for gname, ginfo in PROCESS_GROUPS.items():
        if f in ginfo['features']:
            return ginfo['order'], gname, ginfo['color']
    return 99, 'Other', '#333333'

ranked_feats = sorted(all_feats_set,
                      key=lambda f: (get_process_group(f)[0], -global_max[f]))

n_feats = len(ranked_feats)
feat_labels = [FEAT_DISPLAY.get(f, f) for f in ranked_feats]
feat_cats = [categorize(f) for f in ranked_feats]
feat_group_colors = [get_process_group(f)[2] for f in ranked_feats]
feat_group_names = [get_process_group(f)[1] for f in ranked_feats]

REGIME_BG = {0: '#e8f4fd', 1: '#e8f4fd', 2: '#fff8e1', 3: '#fff8e1', 4: '#fce4ec'}
# Regime assignment per horizon column: R1 (≤10min), R2 (1h–6h), R3 (24h)
REGIME_IDX   = [1, 1, 1, 2, 3]   # per horizon: 5min/10min/1h=regime1, 6h=regime2, 24h=regime3
# REGIME_COLOR / REGIME_NAMES come from fig_config (single source of truth)
REGIME_BG_COL = {1: '#e8f4fd', 2: '#fff8e1', 3: '#fce4ec'}
REGIME_LABEL = {1: 'R1', 2: 'R2', 3: 'R3'}

# ── Figure ──
fig = plt.figure(figsize=(FULL_W, 5.55))  # trimmed 2026-09-04

# Layout: 2 rows — Panel A (heatmap) + Panel B (stacked bars)
_GAP_UP_IN = 0.150   # gaps 2026-10-04: panel (b) block moves up 0.150 in (row gap + b) title lift); both row heights unchanged in inches
_HS0 = HSPACE_PANELS + 0.10   # gaps 2026-10-04: original hspace
_ROWS_IN = (MARGIN_TOP - MARGIN_BOTTOM) * 5.55 / (1 + _HS0 / 2)   # gaps 2026-10-04: summed row heights (in), kept
_HS_NEW = 2 * (_HS0 * _ROWS_IN / 2 - _GAP_UP_IN) / _ROWS_IN   # gaps 2026-10-04: hspace for a gap 0.150 in smaller
gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5.5, 1.30],   # (b) flattened 2026-09-04
                             hspace=_HS_NEW)   # room for the b) title row (2026-09-15); gaps 2026-10-04: was HSPACE_PANELS + 0.10
gs_heat = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_main[0],
                                            wspace=WSPACE_PANELS,
                                            width_ratios=[1, 1, 1, 1, 1, 0.08])
gs_bars = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_main[1],
                                            wspace=WSPACE_PANELS,
                                            width_ratios=[1, 1, 1, 1, 1, 0.08])
fig.subplots_adjust(left=0.115, top=MARGIN_TOP, bottom=MARGIN_BOTTOM + _GAP_UP_IN / 5.55, right=0.90)  # gaps 2026-10-04: bottom raised with (b); left widened so the
# feature row-labels + ylabel stay INSIDE the canvas (tight-crop width == FULL_W standard); right for colorbar

# ═══ Build importance matrices ═══
matrices = {}
for m in ALL_MODELS:
    mat = np.zeros((n_feats, len(HORIZONS)))
    for j, hz in enumerate(HORIZONS):
        s = get_importance(m, hz)
        for i, f in enumerate(ranked_feats):
            mat[i, j] = s.get(f, 0)  # keep negatives!
    matrices[m] = mat

# ═══ Build significance matrices ═══
sig_matrices = {}
for m in ALL_MODELS:
    sig_mat = np.zeros((n_feats, len(HORIZONS)), dtype=bool)
    for j, hz in enumerate(HORIZONS):
        for i, f in enumerate(ranked_feats):
            sig_mat[i, j] = is_significant(m, hz, f)
    sig_matrices[m] = sig_mat

n_sig = sum(sig_matrices[m].sum() for m in ALL_MODELS)
n_total = n_feats * len(HORIZONS) * len(ALL_MODELS)
print(f'Significant: {n_sig}/{n_total} cells (95% CI lower bound > {SIG_THRESHOLD} °C, n=10 stations)')

DOT_SIZE = 9.6  # 12 * 0.8 — reduced 20%

# Diverging colormap: blue (negative) → white (zero) → red (positive)
# P95 of positive values sets the scale; symmetric around zero
all_vals = np.concatenate([mat.ravel() for mat in matrices.values()])
vmax = np.percentile(all_vals[all_vals > 0], 95)
vmin_data = all_vals.min()

# Contour bands: negative side (blue) + positive side (red), white at zero
POS_STEP = 0.04  # positive bands every 0.04 °C
N_NEG = 7         # number of negative bands
neg_bounds = list(np.linspace(vmin_data, 0, N_NEG + 1))
pos_bounds = list(np.arange(POS_STEP, vmax + POS_STEP, POS_STEP))
pos_bounds = [b for b in pos_bounds if b <= vmax + 1e-9]
if pos_bounds[-1] < vmax:
    pos_bounds.append(vmax)
N_POS = len(pos_bounds)  # auto-calculated from step size
bounds = neg_bounds + pos_bounds
print(f'Positive bands: {N_POS} (step={POS_STEP})')

# Colors: blue (negative) → white (zero) → red (positive)
# Sample from RdBu_r: 0.0=blue, 0.5=white, 1.0=red
rdbu_r = plt.cm.RdBu_r
neg_colors = [rdbu_r(i / (2 * N_NEG)) for i in range(N_NEG)]  # blue side (0 → 0.5)
pos_colors = [rdbu_r(0.5 + 0.5 * (i / N_POS)) for i in range(N_POS)]  # white→red (0.5 → 1.0)
colors_list = neg_colors + pos_colors
cmap = mcolors.ListedColormap(colors_list)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

print(f'Value range: {vmin_data:.4f} to {all_vals.max():.4f}, vmax(P95)={vmax:.4f}')
print(f'Negative cells: {(all_vals < 0).sum()}/{len(all_vals)}')

# ═══ Panel A: Heatmaps grouped by HORIZON (5 columns = 5 horizons) ═══
# Each column has 5 model sub-columns inside
heat_axes = []
MODEL_SHORT_LABELS = ['PT', 'iT', 'LS', 'AL', 'DLin']  # compact model labels; DLin, not DL: DL = deep learning in the text (review 2026-09-22)

for idx, hz in enumerate(HORIZONS):
    ax = fig.add_subplot(gs_heat[idx])
    heat_axes.append(ax)

    n_models = len(ALL_MODELS)

    # Set axis limits: x = models, y = features
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(n_feats - 0.5, -0.5)

    # Thin light grid lines
    for i in range(n_feats + 1):
        ax.axhline(i - 0.5, color='#CCCCCC', linewidth=0.5, zorder=3)
    for j in range(n_models + 1):
        # Thick separator before DLinear (index 4)
        if j == len(ALL_MODELS) - 1:
            ax.axvline(j - 0.5, color='#444444', linewidth=0.5, zorder=5)
        else:
            ax.axvline(j - 0.5, color='#CCCCCC', linewidth=0.5, zorder=3)

    # Regime tinting for entire column based on horizon
    r = REGIME_IDX[idx]
    ax.set_facecolor('white')   # regime background tint removed (user decision 2026-06-11, matches Fig 3)

    # Circle size from axes geometry
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cell_w_inch = bbox.width / n_models
    cell_h_inch = bbox.height / n_feats
    cell_px = min(cell_w_inch, cell_h_inch) * fig.dpi
    CIRCLE_FRAC = 0.90
    circle_diameter_pt = cell_px * CIRCLE_FRAC * (72.0 / fig.dpi)
    CIRCLE_S = (circle_diameter_pt ** 2) * (np.pi / 4)
    DOT_FRAC = 0.35
    dot_diameter_pt = cell_px * DOT_FRAC * (72.0 / fig.dpi)
    DOT_S = (dot_diameter_pt ** 2) * (np.pi / 4)

    # Draw circles: x = model index, y = feature index
    for mi, model in enumerate(ALL_MODELS):
        mat = matrices[model]
        sig_mat = sig_matrices[model]
        for i in range(n_feats):
            val = mat[i, idx]  # idx = horizon index
            color = cmap(norm(val))
            ax.scatter(mi, i, s=CIRCLE_S, c=[color], edgecolors='none',
                       linewidths=0, zorder=4)
            if sig_mat[i, idx]:
                ax.scatter(mi, i, s=DOT_S, color='#222', edgecolors='#222',
                           linewidths=0.3, zorder=6, alpha=0.8)

    # ScalarMappable for colorbar
    im = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    im.set_array([])

    # X-axis: model labels at top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(MODEL_SHORT_LABELS, fontsize=FS_DENSE_SM, rotation=0, ha='center')
    ax.tick_params(axis='x', length=0, pad=2, top=True, bottom=False)
    # Model labels in black (consistent with Fig 5)
    for ti, tl in enumerate(ax.get_xticklabels()):
        tl.set_color('black')

    # Y-axis: features (only on leftmost panel)
    ax.set_yticks(np.arange(n_feats))
    if idx == 0:
        ax.set_yticklabels(feat_labels, fontsize=FS_DENSE)
        RAW_FEATS = {'RH'}
        for tick_idx, tick_label in enumerate(ax.get_yticklabels()):
            tick_label.set_color(feat_group_colors[tick_idx])
            # (bold weight on the raw RH label retired 2026-09-22: it was keyed nowhere)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)

    # Horizon title
    ax.set_title(HZ_LABELS_LIST[idx], fontsize=FS_TICK, fontweight='bold', pad=8)

    # Panel label on leftmost
    if idx == 0:
        from matplotlib.transforms import offset_copy as _oc   # project-standard title, lifted clear of the horizon headers (author 2026-09-15)
        ax.text(-0.50, 1.0, 'a) Permutation importance per feature', fontsize=PANEL_LABEL_SIZE,
                fontweight='bold', transform=_oc(ax.transAxes, fig=fig, x=0, y=28.15, units='points'), va='baseline', ha='left')   # gaps 2026-10-04: lift 34 -> 28.15 pt (title lift 0.08 in)
        _TITLE_X = fig.transFigure.inverted().transform(ax.transAxes.transform((-0.50, 1.0)))[0]   # shared left edge for the b) title

    # Spine styling
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(0.2)

    # (stacked bars drawn in Panel B below)

# ═══ Panel B: Stacked bar charts (proper axes) ═══
# Compute global max for shared y-axis
_global_max_grp = 0
for _hz_idx in range(len(HORIZONS)):
    for _model in ALL_MODELS:
        _total = 0
        for _gname in PROCESS_GROUPS:
            for _f_idx, _f in enumerate(ranked_feats):
                if _f in PROCESS_GROUPS[_gname]['features']:
                    _total += max(0, matrices[_model][_f_idx, _hz_idx])
        _global_max_grp = max(_global_max_grp, _total)

MODEL_SHORT_LABELS = ['PT', 'iT', 'LS', 'AL', 'DLin']
BEST_MODEL_PER_HZ = {0: 2, 1: 1, 2: 2, 3: 1, 4: 3}

bar_axes = []
ax_bar0 = None
for pidx, hz in enumerate(HORIZONS):
    if pidx == 0:
        ax_b = fig.add_subplot(gs_bars[pidx])
        ax_bar0 = ax_b
    else:
        ax_b = fig.add_subplot(gs_bars[pidx], sharey=ax_bar0)
    bar_axes.append(ax_b)

    x_pos = np.arange(len(ALL_MODELS))

    for mi, model in enumerate(ALL_MODELS):
        # Compute group sums
        grp_sums_mi = {}
        total_imp = 0
        for gname in PROCESS_GROUPS:
            gs_val = sum(max(0, matrices[model][f_idx, pidx])
                         for f_idx, f in enumerate(ranked_feats)
                         if f in PROCESS_GROUPS[gname]['features'])
            grp_sums_mi[gname] = gs_val
            total_imp += gs_val
        dominant_grp = max(grp_sums_mi, key=lambda g: grp_sums_mi[g]) if total_imp > 0 else None

        # Stacked bars (bottom to top: Moisture → ... → Thermal inertia)
        bottom = 0
        grp_bars = {}
        for gname in reversed(list(PROCESS_GROUPS)):
            val = grp_sums_mi[gname]
            ax_b.bar(mi, val, bottom=bottom, width=0.7,
                     color=PROCESS_GROUPS[gname]['color'], edgecolor='white',
                     linewidth=0.2, alpha=0.85)
            grp_bars[gname] = (bottom, val)
            bottom += val

        # Σ importance label inside the dominant segment REMOVED (author 2026-09-15): the bar height carries
        # the total against the y-axis, and no sentence in the paper quotes it; one number per bar.

    # Compute % for x-axis labels
    model_pcts = []
    for mi_tmp in range(len(ALL_MODELS)):
        t_imp = sum(max(0, matrices[ALL_MODELS[mi_tmp]][f_idx, pidx]) for f_idx in range(n_feats))
        if t_imp > 0:
            grp_sums_tmp = {}
            for gn in PROCESS_GROUPS:
                gs_val = sum(max(0, matrices[ALL_MODELS[mi_tmp]][f_idx, pidx])
                             for f_idx, f in enumerate(ranked_feats) if f in PROCESS_GROUPS[gn]['features'])
                grp_sums_tmp[gn] = gs_val
            dom = max(grp_sums_tmp, key=lambda g: grp_sums_tmp[g])
            model_pcts.append(grp_sums_tmp[dom] / t_imp * 100)
        else:
            model_pcts.append(0)

    # Regime label as title
    r = REGIME_IDX[pidx]
    ax_b.set_title(f'{HZ_LABELS_LIST[pidx]}\n{REGIME_NAMES[r]}',
                    fontsize=FS_DENSE_SM, fontstyle='normal',
                    color='black', pad=3)   # regular black (author 2026-09-15; was regime-colored italic)

    # X-axis labels: model name + % below
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(MODEL_SHORT_LABELS, fontsize=FS_DENSE_SM)
    for mi, tl in enumerate(ax_b.get_xticklabels()):
        if mi == BEST_MODEL_PER_HZ.get(pidx, -1):
            tl.set_fontweight('bold')
            tl.set_color('#000')
        else:
            tl.set_color('#555')
    ax_b.tick_params(axis='x', length=0)
    # % on top of each bar (colored by dominant group)
    for mi in range(len(ALL_MODELS)):
        t_imp = sum(max(0, matrices[ALL_MODELS[mi]][f_idx, pidx]) for f_idx in range(n_feats))
        # Find dominant group color for this model
        if t_imp > 0:
            grp_sums_col = {}
            for gn in PROCESS_GROUPS:
                gs_val = sum(max(0, matrices[ALL_MODELS[mi]][f_idx, pidx])
                             for f_idx, f in enumerate(ranked_feats) if f in PROCESS_GROUPS[gn]['features'])
                grp_sums_col[gn] = gs_val
            dom_col = max(grp_sums_col, key=lambda g: grp_sums_col[g])
            pct_color = PROCESS_GROUPS[dom_col]['color']
        else:
            pct_color = '#555'
        ax_b.text(mi, t_imp + _global_max_grp * 0.02, f'{model_pcts[mi]:.0f}%',
                  ha='center', va='bottom', fontsize=FS_DENSE_SM, color=pct_color,   # dominant-group color: links the % to its segment (author 2026-09-15)
                  fontweight='bold', clip_on=False)

    ax_b.set_ylim(0, _global_max_grp * 1.15)
    ax_b.spines['top'].set_visible(False)        # open style (match Figs 3-4, 2026-06-11)
    ax_b.spines['right'].set_visible(False)
    if pidx == 0:
        ax_b.set_ylabel('Σ Importance\n(°C)', fontsize=FS_DENSE_SM, rotation=0,
                          labelpad=25, va='center', color='#555', fontstyle='italic')
        ax_b.tick_params(axis='y', labelsize=FS_DENSE_SM)
        from matplotlib.transforms import offset_copy as _oc, blended_transform_factory as _btf
        ax_b.text(_TITLE_X, 1.0, 'b) Importance by feature group', fontsize=PANEL_LABEL_SIZE, fontweight='bold',
                  transform=_btf(fig.transFigure, _oc(ax_b.transAxes, fig=fig, x=0, y=21.32, units='points')), va='baseline', ha='left')   # x aligned with a); gaps 2026-10-04: lift 26 -> 21.32 pt (title lift 0.08 in)
    else:
        ax_b.tick_params(axis='y', labelleft=False)

# ═══ Colored labels + separators + group brackets ═══
ax0 = heat_axes[0]

# Find group boundaries (5 groups)
group_ranges = {}
for gname in PROCESS_GROUPS:
    rows = [i for i, g in enumerate(feat_group_names) if g == gname]
    if rows:
        group_ranges[gname] = (min(rows), max(rows))

# Thick separator lines between groups on ALL panels
boundaries = set()
for gname, (r_min, r_max) in group_ranges.items():
    boundaries.add(r_min - 0.5)
    boundaries.add(r_max + 0.5)
boundaries.discard(-0.5)
boundaries.discard(n_feats - 0.5)
for ax in heat_axes:
    for b in boundaries:
        ax.axhline(b, color='#444444', linewidth=0.8, zorder=7)

# (bracket lines removed — legend at bottom instead)

import matplotlib.patches as mpatches

# Vertical colorbar to the right of Panel A
cax = fig.add_subplot(gs_heat[5])
tick_positions, tick_labels, tick_bold_idx = make_cbar_ticks(vmin_data, vmax)
cbar = fig.colorbar(im, cax=cax, orientation='vertical',
                    ticks=tick_positions, spacing='uniform')
cbar.set_label('Mean permutation importance (°C)', fontsize=FS_DENSE, labelpad=3)
cbar.ax.tick_params(labelsize=FS_DENSE_SM)
# two-decimal labels, uniform regular weight (user decision 2026-06-11)
cbar.ax.set_yticklabels([f'{p:.2f}'.replace('-', '\u2212') for p in tick_positions],   # true minus (2026-10-07)
                        fontsize=FS_DENSE_SM, fontweight='normal')




# Annotation: methodology note
# Feature group legend at the bottom
group_legend_handles = []
for gname in PROCESS_GROUPS:
    short = gname.replace('\n', ' ')
    color = PROCESS_GROUPS[gname]['color']
    n = len(PROCESS_GROUPS[gname]['features'])
    group_legend_handles.append(
        mpatches.Patch(facecolor=color, edgecolor='none', label=f'{short} ({n})')
    )
# Legend BELOW the figure, captured by bbox_inches='tight'. Earlier
# (loc='lower center', bbox_to_anchor=(0.50, 0.01)) had its lower-center
# 1% above fig-bottom, but panel (b) sits at fig-bottom (MARGIN_BOTTOM
# =0.02), so the legend's body extended INTO (b)'s bars — visible overlap
# at the 10 min / 1 h horizons. Anchoring the legend's TOP just below
# fig-y 0 with loc='upper center' moves it cleanly out of panel (b);
# tight bbox expands the saved PNG to include it.
fig.legend(handles=group_legend_handles, loc='upper center',
           bbox_to_anchor=(0.50, -0.02 + (_GAP_UP_IN + 0.070833) / 5.55), ncol=3, fontsize=FS_DENSE,   # gaps 2026-10-04: follows (b) up + legend gap 0.08 in
           frameon=True, framealpha=0.9, edgecolor='#cccccc',
           title='Feature groups', title_fontsize=FS_DENSE)
fig.legends[-1].get_frame().set_linewidth(0.4)   # figure-wide legend frame: #cccccc, 0.4 pt (2026-09-17)

# Annotation at bottom of Panel A
fig.canvas.draw()
panel_a_bottom = heat_axes[0].get_position().y0
fig.text(0.50, panel_a_bottom - 0.01,
         'Circle: mean permutation importance (MAE increase, \u00b0C, n = 10 stations). '
         ' \u25cf = significant (95% CI lower > 0.02 \u00b0C).',
         ha='center', va='top', fontsize=FS_DENSE_SM, color='#555',
         fontstyle='italic')

from fig_config import savefig as _savefig
_savefig(plt, 'fig05_feature_importance', bbox='fullwidth')  # exact FULL_W width for consistent print scale
plt.close()
