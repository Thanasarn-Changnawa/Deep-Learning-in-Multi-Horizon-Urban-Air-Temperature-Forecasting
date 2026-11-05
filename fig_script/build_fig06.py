"""Standalone builder for Fig 6: mechanism.
Synced from generate_figures.ipynb cell 13.

Left: dot matrices at 10min vs 6h. Right: bar charts showing best model feature importance.
Both stations are LCZ 5 -- shows surface cover effect on feature usage.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from fig_config import (
    PANEL_LABEL_SIZE,
    FULL_W, DPI, MODEL_COLORS, REGIME_COLOR, HORIZONS, HZ_LABELS,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE, FS_DENSE_SM,
    HSPACE_PANELS, WSPACE_PANELS, WSPACE_CBAR, FEATURE_GROUP_COLORS,
    make_cbar_ticks,
    MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT,
    apply_rcparams,
)
apply_rcparams()

# ── Data ──
imp = pd.read_csv('multihorizon_importance.csv')
ps_df = pd.read_csv('multistation_perstation_metrics.csv')

ALL_MODELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM', 'DLinear']
MODEL_SHORT_LABELS = ['PT', 'iT', 'LS', 'AL', 'DLin']   # DLin, not DL: DL = deep learning in the text (review 2026-09-22)
MODEL_DISPLAY = {'PatchTST': 'PatchTST', 'iTransformer': 'iTransformer',
                 'LSTM': 'LSTM', 'Attention-LSTM': 'Attn-LSTM', 'DLinear': 'DLinear'}
CAT_COLORS = {'Temperature': '#1f77b4', 'Humidity': '#2ca02c', 'Temporal': '#ff7f0e'}

FEAT_DISPLAY = {
    'Temp_rmean_12': 'Temp 1h mean', 'Temp_rmean_72': 'Temp 6h mean',
    'Temp_rmean_288': 'Temp 24h mean',
    'Temp_lag12': 'Temp lag-1h', 'Temp_lag24': 'Temp lag-2h',
    'Temp_lag72': 'Temp lag-6h', 'Temp_lag144': 'Temp lag-12h',
    'Temp_lag288': 'Temp lag-24h',
    'Temp_roc': 'Temp RoC', 'Temp_roc_12': 'Temp RoC-1h',
    'Temp_rstd_12': 'Temp 1h std', 'Temp_rstd_72': 'Temp 6h std',
    'Temp_rstd_288': 'Temp 24h std',
    'Temp_rmax_288': 'Temp 24h max', 'Temp_rmin_288': 'Temp 24h min',
    'Temp_dtr_288': 'Temp DTR-24h', 'Temp': 'Temp (raw)',
    'RH_rmean_12': 'RH 1h mean', 'RH_rmean_72': 'RH 6h mean',
    'RH_rmean_288': 'RH 24h mean',
    'RH_lag12': 'RH lag-1h', 'RH_lag24': 'RH lag-2h',
    'RH_lag72': 'RH lag-6h', 'RH_lag144': 'RH lag-12h',
    'RH_lag288': 'RH lag-24h', 'RH_roc': 'RH RoC', 'RH': 'RH (raw)',
    'hour_sin': 'Hour (sin)', 'hour_cos': 'Hour (cos)',
    'doy_sin': 'DoY (sin)', 'doy_cos': 'DoY (cos)',
}

def categorize(f):
    if any(k in f for k in ['hour_', 'doy_', 'month_']): return 'Temporal'
    if f.startswith('RH') or f == 'RH': return 'Humidity'
    return 'Temperature'

def get_importance(model, horizon):
    sub = imp[(imp['Model'] == model) & (imp['Horizon'] == horizon)]
    return sub.groupby('Feature')['Importance'].mean()

# ── Build shared feature order (same as Fig 7) ──
all_feats_set = set()
for m in ALL_MODELS:
    for hz in HORIZONS:
        all_feats_set |= set(get_importance(m, hz).index)

global_max = {}
for f in all_feats_set:
    vals = [get_importance(m, hz).get(f, 0) for m in ALL_MODELS for hz in HORIZONS]
    global_max[f] = max(vals)

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

ranked_feats = sorted(all_feats_set, key=lambda f: (get_process_group(f)[0], -global_max[f]))
n_feats = len(ranked_feats)
feat_labels = [FEAT_DISPLAY.get(f, f) for f in ranked_feats]
feat_cats = [categorize(f) for f in ranked_feats]
feat_group_colors = [get_process_group(f)[2] for f in ranked_feats]
feat_group_names = [get_process_group(f)[1] for f in ranked_feats]

# Group boundaries (5 groups)
boundaries = []
prev_grp = feat_group_names[0]
for i, g in enumerate(feat_group_names):
    if g != prev_grp:
        boundaries.append(i - 0.5)
        prev_grp = g

# Regime styling
REGIME_IDX = [1, 1, 1, 2, 3]
REGIME_BG_COL = {1: '#e8f4fd', 2: '#fff8e1', 3: '#fce4ec'}
REGIME_NAME = {1: 'Thermal inertia', 2: 'Diurnal decoupling', 3: 'Full diurnal cycle'}

# Colormap (same as Fig 7)
SIG_THRESHOLD = 0.02  # practical significance: CI lower bound must exceed this (°C)
matrices = {}
sig_matrices = {}
for model in ALL_MODELS:
    mat = np.zeros((n_feats, len(HORIZONS)))
    sig_mat = np.zeros((n_feats, len(HORIZONS)), dtype=bool)
    for hi, hz in enumerate(HORIZONS):
        s = get_importance(model, hz)
        for fi, feat in enumerate(ranked_feats):
            mat[fi, hi] = s.get(feat, 0)
            sub = imp[(imp['Model'] == model) & (imp['Horizon'] == hz) & (imp['Feature'] == feat)]
            vals = sub['Importance'].values
            if len(vals) >= 3:
                mean_v = np.mean(vals)
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                sig_mat[fi, hi] = mean_v > 0 and (mean_v - 1.96 * se) > SIG_THRESHOLD
    matrices[model] = mat
    sig_matrices[model] = sig_mat

# Colormap based on per-station values (Baker + Canopied at 10min + 6h)
_station_vals = []
for _sid in [10635585, 10635581]:  # Baker, Canopied
    for _hz in ['10min', '6h']:
        for _m in ALL_MODELS:
            _sub = imp[(imp['Model'] == _m) & (imp['Horizon'] == _hz) & (imp['Station'] == _sid)]
            if len(_sub) > 0:
                _station_vals.extend(_sub['Importance'].values)
_station_vals = np.array(_station_vals)
vmax = np.percentile(_station_vals[_station_vals > 0], 95)
vmin_data = _station_vals.min()

POS_STEP = 0.04  # positive bands every 0.04 °C
N_NEG = 7         # number of negative bands
neg_bounds = list(np.linspace(vmin_data, 0, N_NEG + 1))
pos_bounds = list(np.arange(POS_STEP, vmax + POS_STEP, POS_STEP))
pos_bounds = [b for b in pos_bounds if b <= vmax + 1e-9]
if pos_bounds[-1] < vmax:
    pos_bounds.append(vmax)
N_POS = len(pos_bounds)
bounds = neg_bounds + pos_bounds

rdbu_r = plt.cm.RdBu_r
neg_colors = [rdbu_r(i / (2 * N_NEG)) for i in range(N_NEG)]
pos_colors = [rdbu_r(0.5 + 0.5 * (i / N_POS)) for i in range(N_POS)]
colors_list = neg_colors + pos_colors
cmap = mcolors.ListedColormap(colors_list)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

tick_positions, tick_labels, tick_bold_idx = make_cbar_ticks(vmin_data, vmax)

# ── Figure ──
fig = plt.figure(figsize=(FULL_W, 8.0))  # trimmed 2026-09-04: inter-row gap compressed, matrix rows unchanged
_ROW_CUT_IN = 0.2267   # gaps 2026-10-04: lift the c)/d) row by this many inches (row gap -> 0.12 in)
_ROW_H_IN = 8.0 * (0.90 - 0.04) / (2 + max(HSPACE_PANELS, 0.40))   # gaps 2026-10-04: row height in inches, kept exactly
fig.subplots_adjust(left=0.10, right=MARGIN_RIGHT, top=0.90, bottom=0.04 + _ROW_CUT_IN / 8.0)   # gaps 2026-10-04: bottom raised, top row fixed

# Layout: 2 rows — Panel A (Baker/Non-veg), Panel B (Canopied/Veg)
gs_main = gridspec.GridSpec(2, 1, figure=fig, hspace=(max(HSPACE_PANELS, 0.40) * _ROW_H_IN - _ROW_CUT_IN) / _ROW_H_IN,   # gaps 2026-10-04: inter-row gap cut by _ROW_CUT_IN, rows keep their inch height
                            height_ratios=[1, 1])

DL_MODELS_C = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']

# Compute global max importance across both stations for shared bar scale
_global_xmax = 0
for _sid in [10635585, 10635581]:
    for _hz in ['10min', '6h']:
        for _m in DL_MODELS_C:
            _sub = imp[(imp['Model'] == _m) & (imp['Horizon'] == _hz) & (imp['Station'] == _sid)]
            if len(_sub) > 0:
                _global_xmax = max(_global_xmax, _sub['Importance'].max())


def draw_station_panel(gs_row, sid, station_label, sc_color, panel_label_left, panel_label_right, row_title):
    """Draw left dot matrices (10min, 6h) + right bar charts for one station."""
    gs_lr = gs_row.subgridspec(1, 2, wspace=WSPACE_CBAR, width_ratios=[1, 1])
    _rp = gs_row.get_position(fig)
    _lp = gs_lr[0].get_position(fig)
    _mx1 = _lp.x0 + (_lp.x1 - _lp.x0) * (2.0 / 2.06)   # right edge of the two matrices (excl. cbar col)
    # centered two-line station header REMOVED (2026-09-15): the station now lives in the a)-d) panel titles (project standard)
    _row_title = row_title.replace(chr(10), ', ')   # 'Built (LCZ 5), Baker Building' -> used to build the panel titles below

    # ── LEFT: dot matrices ──
    gs_det = gs_lr[0].subgridspec(1, 3, wspace=WSPACE_PANELS, width_ratios=[1, 1, 0.06])
    hz_pairs = [('10min', '10 min'), ('6h', '6 h')]

    for hi, (hz_detail, hz_lbl) in enumerate(hz_pairs):
        ax_d = fig.add_subplot(gs_det[hi])
        hz_idx = HORIZONS.index(hz_detail)
        n_models = len(ALL_MODELS)

        ax_d.set_xlim(-0.5, n_models - 0.5)
        ax_d.set_ylim(n_feats - 0.5, -0.5)

        for i in range(n_feats + 1):
            ax_d.axhline(i - 0.5, color='#CCCCCC', linewidth=0.5, zorder=3)
        for j in range(n_models + 1):
            if j == len(ALL_MODELS) - 1:
                ax_d.axvline(j - 0.5, color='#444444', linewidth=0.5, zorder=5)
            else:
                ax_d.axvline(j - 0.5, color='#CCCCCC', linewidth=0.5, zorder=3)
        for b in boundaries:
            ax_d.axhline(b, color='#444444', linewidth=0.5, zorder=7)

        r = REGIME_IDX[hz_idx]
        ax_d.set_facecolor('white')   # regime tint removed (user decision 2026-06-11, matches Figs 3,5)

        fig.canvas.draw()
        bbox_d = ax_d.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cell_px_d = min(bbox_d.width / n_models, bbox_d.height / n_feats) * fig.dpi
        circle_s_d = ((cell_px_d * 0.88 * (72.0 / fig.dpi)) ** 2) * (np.pi / 4)

        # Find best model (lowest MAE)
        model_maes = []
        for model in ALL_MODELS:
            mr = ps_df[(ps_df['Station'] == sid) & (ps_df['Horizon'] == hz_detail)
                       & (ps_df['Model'] == model)]
            model_maes.append(float(mr['MAE'].values[0]) if len(mr) > 0 else 999)
        best_mi = int(np.argmin(model_maes))

        for mi, model in enumerate(ALL_MODELS):
            for i, feat in enumerate(ranked_feats):
                sub = imp[(imp['Model'] == model) & (imp['Horizon'] == hz_detail)
                          & (imp['Station'] == sid) & (imp['Feature'] == feat)]
                val = sub['Importance'].values[0] if len(sub) > 0 else 0
                c = cmap(norm(val))
                ax_d.scatter(mi, i, s=circle_s_d, c=[c], edgecolors='none',
                             linewidths=0, zorder=4)

        # MAE values below, bold winner
        for mi2 in range(len(ALL_MODELS)):
            mae_val = model_maes[mi2]
            is_best = (mi2 == best_mi)
            fw = 'bold' if is_best else 'normal'
            col = 'black' if is_best else '#888'
            txt = f'{mae_val:.3f}' if mae_val < 999 else '\u2014'
            ax_d.text(mi2, n_feats + 0.3, txt, ha='center', va='top',
                      fontsize=FS_DENSE_SM, fontweight=fw, color=col, clip_on=False)

        if hi == 1:   # station line under the MAE footer, centered on the two matrices (author 2026-09-15)
            _station_name = _row_title.split(', ', 1)[1]
            _lp2 = gs_det[0].get_position(fig); _rp2 = gs_det[1].get_position(fig)
            _y_foot = fig.transFigure.inverted().transform(ax_d.transData.transform((0, n_feats + 0.3)))[1]
            fig.text((_lp2.x0 + _rp2.x1) / 2, _y_foot - 14 / 72.0 / fig.get_figheight(), _station_name,
                     ha='center', va='top', fontsize=FS_LABEL, fontweight='bold', color='black')
        ax_d.xaxis.set_ticks_position('top')
        ax_d.xaxis.set_label_position('top')
        ax_d.set_xticks(np.arange(n_models))
        ax_d.set_xticklabels(MODEL_SHORT_LABELS, fontsize=FS_DENSE_SM)
        for tii, tl in enumerate(ax_d.get_xticklabels()):
            tl.set_color('black')
        ax_d.tick_params(axis='x', length=0, pad=2)

        ax_d.set_yticks(np.arange(n_feats))
        if hi == 0:
            ax_d.set_yticklabels(feat_labels, fontsize=FS_DENSE)
            for tick_idx, tick_label in enumerate(ax_d.get_yticklabels()):
                tick_label.set_color(feat_group_colors[tick_idx])
        else:
            ax_d.set_yticklabels([])
        ax_d.tick_params(axis='y', length=0)

        parts = station_label.split(' \u2192 ')
        if len(parts) == 3:
            line1 = f'{parts[0]} \u2192 {parts[1]}'
            line2 = parts[2]
        else:
            line1 = station_label
            line2 = ''
        _reg_dot = REGIME_NAME[REGIME_IDX[hz_idx]]
        ax_d.set_title(hz_lbl, fontsize=FS_DENSE_SM,   # regime sublabel dropped (author 2026-09-04)
                        fontweight='bold', color='black', pad=4)   # black titles (author 2026-09-04)

        if hi == 0:
            from matplotlib.transforms import offset_copy as _oc, blended_transform_factory as _btf
            _cover, _station = _row_title.split(', ', 1)   # 'Built (LCZ 5)', 'Baker Building'
            _cover_short = _cover.split(' (')[0]           # 'Built' / 'Tree canopy'
            _xl = fig.transFigure.inverted().transform(ax_d.transAxes.transform((-0.55, 1.0)))[0]
            _xl = max(_xl, 0.012)                          # never left of the canvas edge
            ax_d.text(_xl, 1.0, f'{panel_label_left} Importance matrices: {_cover_short}',   # content first, cover last (author 2026-09-15)
                      fontsize=PANEL_LABEL_SIZE, fontweight='bold', color='black',
                      transform=_btf(fig.transFigure, _oc(ax_d.transAxes, fig=fig, x=0, y=23.9, units='points')), va='baseline', ha='left')   # gaps 2026-10-04: title lift 30 -> 23.9 pt (lift band -> 0.08 in)

        for spine in ax_d.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.5)

    # Colorbar for dot matrices
    cax_d = fig.add_subplot(gs_det[2])
    im_tmp = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_d = fig.colorbar(im_tmp, cax=cax_d, orientation='vertical',
                           ticks=tick_positions, spacing='uniform')
    cbar_d.set_label('Per-station permutation\nimportance (\u00b0C)', fontsize=FS_DENSE, labelpad=3)
    cbar_d.ax.tick_params(labelsize=FS_DENSE_SM)
    # two-decimal labels, uniform regular weight (matches Fig 5, 2026-06-11)
    cbar_d.ax.set_yticklabels([f'{p:.2f}'.replace('-', '\u2212') for p in tick_positions],   # true minus (2026-10-07)
                              fontsize=FS_DENSE_SM, fontweight='normal')

    # ── RIGHT: bar charts ──
    gs_bars = gs_lr[1].subgridspec(1, 2, wspace=WSPACE_PANELS)

    for ri, (hz_detail, hz_lbl, regime_lbl) in enumerate([
        ('10min', '10 min', 'Thermal inertia'),
        ('6h', '6 h', 'Diurnal decoupling'),   # canonical regime-2 name (matches body, Figs 3-5)
    ]):
        ax = fig.add_subplot(gs_bars[ri])

        # Find best DL model by MAE
        model_maes = {}
        for model in DL_MODELS_C:
            mr = ps_df[(ps_df['Station'] == sid) & (ps_df['Horizon'] == hz_detail)
                       & (ps_df['Model'] == model)]
            if len(mr) > 0:
                model_maes[model] = mr.iloc[0]['MAE']
        best_model = min(model_maes, key=model_maes.get)

        sub = imp[(imp['Model'] == best_model) & (imp['Horizon'] == hz_detail)
                  & (imp['Station'] == sid)]
        feat_imp = sub.set_index('Feature')['Importance']

        ax.set_ylim(n_feats - 0.5, -0.5)

        total_imp = sum(max(0, feat_imp.get(f, 0)) for f in ranked_feats)
        for i, f in enumerate(ranked_feats):
            val = feat_imp.get(f, 0)
            color = get_process_group(f)[2]
            ax.barh(i, val, height=0.7, color=color, edgecolor='none',
                    alpha=0.85, zorder=3)
            # % at end of bar (only for significant bars)
            if val > 0 and total_imp > 0:
                pct = val / total_imp * 100
                if pct >= 5:  # only show if >= 5%
                    ax.text(val + _global_xmax * 0.02, i, f'{pct:.0f}%',
                            ha='left', va='center', fontsize=FS_DENSE_SM,
                            color='#555', clip_on=False)
        ax.axvline(0, color='#444444', linewidth=0.5, zorder=4)

        # horizontal grey lines removed from the bar panels for a cleaner look
        # (user decision 2026-06-11; group structure is carried by the colored y-labels)

        ax.set_yticks(np.arange(n_feats))
        if ri == 0:
            ax.set_yticklabels(feat_labels, fontsize=FS_DENSE_SM)
            for ti, tl in enumerate(ax.get_yticklabels()):
                tl.set_color(feat_group_colors[ti])
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=FS_DENSE_SM)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.grid(True, linestyle='--', linewidth=0.2, alpha=0.4)
        ax.set_axisbelow(True)

        best_short = MODEL_DISPLAY[best_model]
        model_idx = DL_MODELS_C.index(best_model)
        short_lbl = ['PT', 'iT', 'LS', 'AL'][model_idx]

        # Panel label for right side
        if ri == 0:
            from matplotlib.transforms import offset_copy as _oc
            _cover, _station = _row_title.split(', ', 1)
            _cover_short = _cover.split(' (')[0]
            ax.text(-0.15, 1.0, f'{panel_label_right} Best-model bars: {_cover_short}',
                    fontsize=PANEL_LABEL_SIZE, fontweight='bold', color='black',
                    transform=_oc(ax.transAxes, fig=fig, x=0, y=23.9, units='points'), va='baseline', ha='left')   # gaps 2026-10-04: title lift 30 -> 23.9 pt (lift band -> 0.08 in)

        # Colored model label at top center (matching left panel x-tick style)
        ax.text(0.5, 1.01, best_short, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=FS_DENSE_SM, fontweight='bold',
                color='black')   # black model name (author 2026-09-04)

        # Title: station name, horizon, regime (below model label)
        station_short = station_label.split(chr(8594))[-1].strip()
        _reg_idx_bar = {v: k for k, v in REGIME_NAME.items()}[regime_lbl]
        ax.set_title(hz_lbl,   # regime sublabel dropped (author 2026-09-04)
                     fontsize=FS_DENSE, fontweight='bold', color='black', pad=12)   # black titles (author 2026-09-04)
        ax.set_xlim(None, _global_xmax * 1.3)
        ax.set_xlabel('Importance (\u00b0C)', fontsize=FS_DENSE)

        # open style for bar panels (match suite): keep left+bottom, hide top+right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_edgecolor('#888888')
            ax.spines[sp].set_linewidth(0.5)


# Panel A: Baker Building (Non-veg, Built, LCZ 5)
draw_station_panel(gs_main[0], 10635585,
                   'Non-veg \u2192 Built \u2192 Baker Bldg', '#8B4513', 'a)', 'b)',
                   'Built (LCZ 5)\nBaker Building')

# Panel B: Canopied Walkway (Veg, Tree canopy, LCZ 5)
draw_station_panel(gs_main[1], 10635581,
                   'Veg \u2192 Tree canopy \u2192 Canopied Wk', '#2ca02c', 'c)', 'd)',
                   'Tree canopy (LCZ 5)\nCanopied Walkway')

# (LCZ 5 header removed)

# Suptitle
# (suptitle removed -- caption goes below figure per Elsevier style)

# Annotation
# in-figure footnote removed 2026-06-11 (collided with panel d x-label; content is in the caption)

from fig_config import savefig as _savefig
_savefig(plt, 'fig06_mechanism', bbox='fullwidth')  # exact FULL_W width for consistent print scale
plt.close()
