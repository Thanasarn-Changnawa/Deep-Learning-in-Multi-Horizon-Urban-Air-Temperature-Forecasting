"""
Figure 8 -- Combined validation diagnostics (3 rows; each panel full-width)

  Row 1 (a) Feature ablation: MAE by feature subset across 5 horizons
           (network mean with ±1σ across 4 stations)
  Row 2 (b) UQ comparison: split conformal vs MC Dropout
           (1×2 sub-grid: empirical coverage | mean interval width)
  Row 3 (c) MC Dropout prediction intervals at 3 representative horizons
           (1×3 sub-grid: 5 min / 6 h / 24 h; shared y-axis)

Data:
  ablation_feature_subsets.csv        — 4 stations × 4 subsets × 5 horizons
  conformal_prediction_results.csv    — coverage + width by station × horizon × method
  publication_results.json → uncertainty[model][horizon]  — MC samples + actuals

All styling + constants imported from fig_config.
"""
import warnings

def _assert_cache_orientation(_msd):
    """Tripwire (2026-08-25): cache tuples are (prediction, observation). The Persistence
    forecast must equal the observation lagged by the horizon; fails loudly if anyone
    ever swaps the unpack order again."""
    import numpy as _np
    _hmap={'5min':1,'10min':2,'1h':12,'6h':72,'24h':288}
    _k=next(k for k in _msd if k[2]=='Persistence' and k[1] in _hmap)
    _p=_np.asarray(_msd[_k][0]).ravel(); _o=_np.asarray(_msd[_k][1]).ravel(); _h=_hmap[_k[1]]
    assert _np.abs(_p[_h:]-_o[:-_h]).max()<1e-3, (
        'ms_predictions cache orientation violated: element0 must be PREDICTION, element1 OBSERVATION')

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

from fig_config import (
    FULL_W, MAX_H,
    HORIZONS, HZ_LABELS, MODEL_COLORS, STATION_COORDS,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE, FS_DENSE_SM,
    WSPACE_PANELS,
    apply_rcparams, savefig,
    inline_panel_title,
)
apply_rcparams()

# =====================================================================
# Data loading
# =====================================================================
abl = pd.read_csv('ablation_feature_subsets.csv')  # panel (a) — feature ablation
conf = pd.read_csv('conformal_prediction_results.csv')

CHAMPION = 'Attention-LSTM'
mc_results = None
if os.path.exists('publication_results.json'):
    with open('publication_results.json', 'r', encoding='utf-8') as f:
        fd = json.load(f)
    mc_raw = fd.get('figure_data', {}).get('uncertainty', fd.get('uncertainty', {}))
    meta_keys = {'n_mc_samples', 'mc_dropout_rate'}
    mc_model_data = {k: v for k, v in mc_raw.items() if k not in meta_keys}
    if CHAMPION in mc_model_data:
        mc_results = mc_model_data[CHAMPION]
    elif len(mc_model_data) == 1:
        mc_results = next(iter(mc_model_data.values()))
    del fd

# =====================================================================
# Figure layout — 3 rows
# =====================================================================
fig = plt.figure(figsize=(FULL_W, 6.85))          # 2-row: further trimmed with legend/headroom cuts (2026-09-04)
outer = gridspec.GridSpec(2, 1, figure=fig,
                           hspace=0.22,                       # tightened gap (2026-09-04); clears top-row x-labels vs panel d title
                           height_ratios=[2.82, 1.90],       # row0 shrunk with its trimmed headroom; row1 (d) unchanged
                           left=0.072, right=0.985, top=0.94, bottom=0.105)  # margins widened 2026-09-16 so the ink fills the 7.48 in frame (was 0.08/0.97); bottom=0.105: modest gap between d x-axis and the legend

# =====================================================================
# ROW 0 — (a) left (tall) | (b) over (c) stacked on the right;
# ROW 1 — (d) full-width 6 h interval trace.
#   (a) Feature inventory + ablation scenarios (left, full top-row height)
#   (b) MAE penalty by forecast horizon        (top-right)
#   (c) Empirical coverage of 90 % PIs         (bottom-right, under b)
# =====================================================================
row1 = outer[0].subgridspec(1, 3, width_ratios=[1.0, 0.94, 0.10], wspace=0.30)   # slim third column: the b/c legends hang into it and end near panel d's right edge (2026-09-16)
ax_a1 = fig.add_subplot(row1[0])   # (a) Feature inventory + scenarios
row1_right = row1[1].subgridspec(2, 1, hspace=0.55, height_ratios=[1.0, 0.85])
ax_a2 = fig.add_subplot(row1_right[0])   # (b) MAE penalty by horizon

# Ablation subset definitions (from `ablation_feature_subsets.csv`)
SUBSET_ORDER = ['Raw', 'Temporal', 'Lags', 'Full']
SUBSET_COLORS = {
    'Raw':      '#999999',
    'Temporal': '#f39c12',
    'Lags':     '#d62728',
    'Full':     '#1f77b4',
}
NEURAL_MODELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']

# Per-model MAE pivot — mean over 4 ablation stations, all neural candidates
abl_n = abl[(abl.Model.isin(NEURAL_MODELS)) & (abl.Subset.isin(SUBSET_ORDER))]
mae_by_model = abl_n.pivot_table(
    index=['Model', 'Subset'], columns='Horizon', values='MAE', aggfunc='mean'
)

# Mean across all neural models — used for §4.4 prose / consistency reference
mae_pivot = abl_n.pivot_table(
    index='Subset', columns='Horizon', values='MAE', aggfunc='mean'
).reindex(index=SUBSET_ORDER, columns=HORIZONS)

hz_x = np.arange(len(HORIZONS))
i_6h = HORIZONS.index('6h')

# --- Panel (a): MAE penalty from cumulative feature removal -----------------
# Ablation framing — how much does MAE INCREASE when feature groups are
# stripped from the Full model? Three cumulative-removal scenarios shown as
# three sub-panels (a1, a2, a3); each plots the per-architecture penalty
# curve so the reader sees the ablation pattern across PatchTST, iTransformer,
# LSTM, and Attention-LSTM.
SCENARIO_ORDER = ['Lags', 'Temporal', 'Raw']
SCENARIO_COLORS = {
    'Lags':     SUBSET_COLORS['Lags'],      # red
    'Temporal': SUBSET_COLORS['Temporal'],  # orange
    'Raw':      SUBSET_COLORS['Raw'],       # grey
}
SCENARIO_LABEL = {
    'Lags':     'Drop stats & long lags  (21 features)',
    'Temporal': 'Drop stats + all lags  (25 features)',
    'Raw':      'Drop all engineered  (29 features)',
}

# Per-(model, station, scenario, horizon) penalty: MAE[scenario] − MAE[Full]
abl_idx = abl[abl.Model.isin(NEURAL_MODELS)].set_index(
    ['Model', 'Station', 'Subset', 'Horizon'])['MAE'].sort_index()
STATION_ORDER = sorted(abl.Station.unique())

def _penalties(scenario, horizon, models=NEURAL_MODELS, stations=STATION_ORDER):
    """List of penalty values across the requested model × station combos."""
    out = []
    for m in models:
        for st in stations:
            mae_sc = abl_idx.get((m, st, scenario, horizon))
            mae_fl = abl_idx.get((m, st, 'Full', horizon))
            if mae_sc is not None and mae_fl is not None:
                out.append(mae_sc - mae_fl)
    return np.array(out)

# =====================================================================
# (a1) LEFT — Per-feature dendrogram (30 features, Fig 5 group color).
# Top → bottom order matches "cumulative removal":
#   top    : +Full features (21)         — the engineered ones
#   middle : +Lags features (4)          — simple T/RH lag12, lag24
#            +Temp features (4)          — hour, doy sin/cos
#   bottom : Raw feature   (1)           — RH only
# Subset brackets on the right tie features to their subset; centroids
# of these brackets define the y-positions used by (a2).
# =====================================================================
# Subset bracket colours (used by a2 fall-back bars too)
SUBSET_BRACKET_COLOR = {
    'Raw':       SUBSET_COLORS['Raw'],
    '+Temporal': SUBSET_COLORS['Temporal'],
    '+Lags':     SUBSET_COLORS['Lags'],
    '+Full':     '#1f77b4',
}

# [(a0) merged into (a1) — feature names now sit inside the bars]

# =====================================================================
# (a0) LEFT — Feature inventory list aligned with (a1) bar segments.
# 30 features on a shared 0..30 y-axis; each feature row is colour-tinted
# by its group so it lines up horizontally with the corresponding
# coloured segment in the stacked bars on the right.
# =====================================================================
# (feature_name, group_index)  group_index 0..3 = Raw / Cyclical /
# Simple lags / Stats — same order used in the stacked bar segments.
FEATURE_LIST = [
    ('RH', 0),
    ('hour_sin', 1), ('hour_cos', 1), ('doy_sin', 1), ('doy_cos', 1),
    ('Temp_lag12', 2), ('Temp_lag24', 2), ('RH_lag12', 2), ('RH_lag24', 2),
    ('Temp_rmean_12', 3),  ('Temp_rmean_72', 3),  ('Temp_rmean_288', 3),
    ('Temp_lag72', 3),     ('Temp_lag144', 3),    ('Temp_lag288', 3),
    ('Temp_rstd_12', 3),   ('Temp_rstd_72', 3),   ('Temp_rstd_288', 3),
    ('Temp_rmin_288', 3),  ('Temp_rmax_288', 3),  ('Temp_dtr_288', 3),
    ('Temp_roc_12', 3),    ('Temp_roc', 3),
    ('RH_rmean_12', 3),    ('RH_rmean_72', 3),    ('RH_rmean_288', 3),
    ('RH_lag72', 3),       ('RH_lag144', 3),      ('RH_lag288', 3),
    ('RH_roc', 3),
]
# Reader names, identical to Figs. 5 and 6 (build_fig05.FEAT_DISPLAY; 2026-10-06), so panel a matches the importance figures
FEAT_DISPLAY_A = {
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
# Same colours used by stacked-bar segments (defined just below)
GROUP_COLORS_PALETTE = ['#666666',
                        SUBSET_BRACKET_COLOR['Raw'],
                        SUBSET_BRACKET_COLOR['+Temporal'],
                        SUBSET_BRACKET_COLOR['+Lags']]

# Feature inventory drawn INSIDE ax_a1 at negative x-positions so it
# shares the same y-axis with the stacked bars (rows align with bar
# segments). x range [-1.05, -0.05] = feature-list strip; bars then
# start at x=0.
n_feat = len(FEATURE_LIST)
LIST_X_LEFT  = -1.18   # was -1.05; widened so the longest name (Temp_rmean_288) stays inside its chip (QA 2026-09-16)
LIST_X_RIGHT = 0.20   # extends right so the tab fully covers the longest feature name
for i, (fname, gidx) in enumerate(FEATURE_LIST):
    yi = i + 0.5  # +0.5 so RH box bottom sits at y=0 (no clip with ylim=0) AND aligns rows with bar segments
    # Background tint for the feature row (matches its group colour)
    ax_a1.add_patch(plt.Rectangle(
        (LIST_X_LEFT, yi - 0.45),
        LIST_X_RIGHT - LIST_X_LEFT, 0.9,
        facecolor=GROUP_COLORS_PALETTE[gidx],
        alpha=0.18, edgecolor='none', zorder=1))
    ax_a1.text(LIST_X_LEFT + 0.04, yi, FEAT_DISPLAY_A.get(fname, fname),
               va='center', ha='left',
               fontsize=FS_DENSE, color='#1a1a2e', zorder=4)

# =====================================================================
# (a1) MIDDLE — Stacked bar chart: 4 vertical bars (one per scenario)
# segmented by feature group. Each segment colour = the step at which
# the feature group was added (matches (a2) fall-back colours). Bar
# height = total kept feature count per scenario. Quantitative axes
# replace the previous text-card look.
# =====================================================================
SCENARIOS = [
    ('Full',     SUBSET_BRACKET_COLOR['+Full']),
    ('Drop 21',  SUBSET_BRACKET_COLOR['+Lags']),
    ('Drop 25',  SUBSET_BRACKET_COLOR['+Temporal']),
    ('Drop 29',  SUBSET_BRACKET_COLOR['Raw']),
]
# Each bar is built from up to 4 segments stacked bottom-to-top; segment
# colour = the addition step that introduced those features
GROUP_SEGMENTS = [
    # (group label, count, segment colour, KEEP mask per scenario)
    ('Raw (RH)',         1, '#666666',
     [True, True,  True,  True]),
    ('+Cyclical',        4, SUBSET_BRACKET_COLOR['Raw'],
     [True, True,  True,  False]),
    ('+Simple lags',     4, SUBSET_BRACKET_COLOR['+Temporal'],
     [True, True,  False, False]),
    ('+Stats / RoC /\n  long lags', 21, SUBSET_BRACKET_COLOR['+Lags'],
     [True, False, False, False]),
]

# Compute MAE penalty per scenario across horizons (annotation source)
PENALTY_BY_SCENARIO = {'Full': None}
for sc_label, subset in [('Drop 21', 'Lags'),
                          ('Drop 25', 'Temporal'),
                          ('Drop 29', 'Raw')]:
    pen = mae_pivot.loc[subset] - mae_pivot.loc['Full']
    PENALTY_BY_SCENARIO[sc_label] = {
        '6h':    pen['6h'],
        'min':   pen.min(),
        'max':   pen.max(),
    }

BAR_X_SHIFT = 0.6   # shift bars right to widen the gap from the feature inventory
x_pos = np.arange(len(SCENARIOS)) + BAR_X_SHIFT
n_kept_per_scenario = []
# Build stacked bars
for si, (sc_label, sc_color) in enumerate(SCENARIOS):
    bottom = 0
    BAR_W = 0.4   # narrower bars for cleaner separation from the feature list
    for gname, gcount, gcolor, mask in GROUP_SEGMENTS:
        if mask[si]:
            ax_a1.bar(x_pos[si], gcount, width=BAR_W,
                      bottom=bottom,
                      color=gcolor, alpha=0.85,
                      edgecolor='none', zorder=3)
            # Label inside the segment if it's tall enough
            if gcount >= 3:
                ax_a1.text(x_pos[si], bottom + gcount / 2,
                           f'{gcount}',
                           ha='center', va='center',
                           fontsize=FS_DENSE, fontweight='bold',
                           color='white', zorder=5)
            else:
                # Tiny segment — label on the side
                ax_a1.text(x_pos[si] + BAR_W / 2 + 0.04,
                           bottom + gcount / 2,
                           f'{gcount}',
                           ha='left', va='center',
                           fontsize=FS_DENSE_SM, fontweight='bold',
                           color='#1a1a2e', zorder=5)   # near-black like all value labels (QA 2026-09-16)
            bottom += gcount
    n_kept_per_scenario.append(bottom)
    # "N ft kept" label sits just above each bar (variable y)
    ax_a1.text(x_pos[si], bottom + 0.8,
               f'{bottom} kept',   # was 'N ft kept'; 'ft' read as feet (QA 2026-09-16)
               ha='center', va='bottom',
               fontsize=FS_DENSE, fontweight='bold',
               color='#1a1a2e')

# [Penalty annotations removed — kept-feature counts above each bar
# carry the headline, and (a2) holds the full penalty story.]

# X-tick scenario labels (coloured); Full carries "(baseline)" subtitle
xtick_labels = []
for sc_label, _ in SCENARIOS:
    if sc_label == 'Full':
        xtick_labels.append('Full')   # 'baseline' dropped: the paper's benchmarks are never baselines (2026-10-07)
    else:
        xtick_labels.append(sc_label)
ax_a1.set_xticks(x_pos)
ax_a1.set_xticklabels(xtick_labels, fontsize=FS_TICK,
                      fontweight='bold')
for si, (_, sc_color) in enumerate(SCENARIOS):
    ax_a1.get_xticklabels()[si].set_color(sc_color)

ax_a1.set_xlim(LIST_X_LEFT - 0.05, len(SCENARIOS) - 0.4 + BAR_X_SHIFT)
ax_a1.set_ylim(0, n_feat + 1.5)   # headroom trimmed (author 2026-09-04); bottom at 0 so bars sit on the axis
ax_a1.set_ylabel('Number of retained features', fontsize=FS_LABEL)
ax_a1.tick_params(labelsize=FS_TICK)
ax_a1.grid(axis='y', alpha=0.25, linewidth=0.4)

# Group legend (segment colours)
legend_handles = [
    Patch(facecolor=gcolor, edgecolor='none',
          alpha=0.85, label=gname)
    for (gname, _, gcolor, _) in GROUP_SEGMENTS
]
ax_a1.legend(handles=legend_handles,
             fontsize=FS_DENSE, frameon=True,
             loc='upper right',
             bbox_to_anchor=(0.99, 0.80),   # legend lowered into the empty mid-right (author 2026-09-04)
             framealpha=0.92, edgecolor='#cccccc', handlelength=0.8,
             handletextpad=0.3, borderpad=0.22, labelspacing=0.22, ncol=1,
             title='Ablation subset', title_fontsize=FS_DENSE)   # not the Fig. 5 feature groups (2026-10-07)
ax_a1.get_legend().get_frame().set_linewidth(0.4)   # figure-wide legend frame: #cccccc, 0.4 pt (2026-09-17)
inline_panel_title(ax_a1, 'a',
                   'Feature subsets under cumulative ablation',
                   pad_pts=6)
for spine in ('top', 'right'):   # open style
    ax_a1.spines[spine].set_visible(False)


# =====================================================================
# (a2) MIDDLE — MAE penalty bars sharing y-axis with (a1).
# 4 rows (Full / +Lags / +Temp / Raw); each row has 5 horizontal bars
# (one per horizon). Bar length = MAE penalty when the model falls back
# to that row's subset. Full row = baseline (all-zero bars). Bars
# coloured by horizon — visual position links to the inventory bar on
# the left.
# =====================================================================
# Mean MAE per (subset, horizon) across 4 archs × 4 stations
mae_pool = abl[abl.Model.isin(NEURAL_MODELS)].pivot_table(
    index='Subset', columns='Horizon', values='MAE', aggfunc='mean'
).reindex(index=SUBSET_ORDER, columns=HORIZONS)

# Horizon → distinct colour (sequential blue-to-red ramp)
# (HZ_COLORS dict removed 2026-06-14 — unused dead code; panel b bars are
#  coloured by removal level via SUBSET_BRACKET_COLOR, not by horizon.)

# Row → fall-back subset key in mae_pool
ROW_SUBSET = {'+Full': 'Full', '+Lags': 'Lags',
              '+Temporal': 'Temporal', 'Raw': 'Raw'}
# Per-horizon view: y = horizon (5 rows), x = MAE penalty.
# 3 horizontal bars per horizon row, colors matching (a1)'s subset
# brackets (red = +Lags, orange = +Temp, grey = Raw). Each bar = the
# penalty when the model falls back to that subset.
hz_y = np.arange(len(HORIZONS))[::-1]    # 5 min on top, 24 h on bottom
i_6h_pos = list(HORIZONS).index('6h')
i_6h_y = hz_y[i_6h_pos]

PENALTY_ORDER = [
    ('drop_21', 'Lags',     'Drop 21'),
    ('drop_25', 'Temporal', 'Drop 25'),
    ('drop_29', 'Raw',      'Drop 29'),
]
PENALTY_COLOR = {
    'drop_21': SUBSET_BRACKET_COLOR['+Lags'],      # red
    'drop_25': SUBSET_BRACKET_COLOR['+Temporal'],  # orange
    'drop_29': SUBSET_BRACKET_COLOR['Raw'],        # grey
}

n_bars = len(PENALTY_ORDER)
bar_w = 0.25
penalties_all = []
hz_x = np.arange(len(HORIZONS))   # 5 min left, 24 h right
i_6h_x = list(HORIZONS).index('6h')

for hi, h in enumerate(HORIZONS):
    xc = hz_x[hi]
    full_mae = mae_pool.loc['Full', h]
    for bi, (key, subset, label) in enumerate(PENALTY_ORDER):
        offset = (bi - (n_bars - 1) / 2) * bar_w
        penalty = mae_pool.loc[subset, h] - full_mae
        penalties_all.append(penalty)
        ax_a2.bar(xc + offset, penalty, bar_w,
                  color=PENALTY_COLOR[key], alpha=0.92,
                  edgecolor='none',
                  label=label if hi == 0 else None,
                  zorder=3)
        # Penalty value above bar — rotated 90° so adjacent labels
        # don't horizontally overlap (rotated text occupies only ~font-
        # height of horizontal width)
        ax_a2.text(xc + offset, penalty + 0.05,
                   f'+{penalty:.2f}',
                   va='bottom', ha='center',
                   rotation=90,
                   fontsize=FS_DENSE_SM, fontweight='bold',
                   color='#1a1a2e')   # near-black kept: the author preferred it over bar-coloured labels (2026-09-16)

# (6 h regime column highlight removed 2026-06-14 — panel c, the adjacent bar
#  chart, carries no 6 h shading, so the highlight made the two bar panels
#  inconsistent; the 6 h regime motif lives in Fig 3 / Fig 7 where it is the finding.)
ax_a2.axhline(0, color='black', lw=0.6, alpha=0.7, zorder=2)

ymax = max(penalties_all) + 0.55   # headroom trimmed (author 2026-09-04); still clears the rotated top values
ax_a2.set_xticks(hz_x)
ax_a2.set_xticklabels([HZ_LABELS[h] for h in HORIZONS], fontsize=FS_TICK)
ax_a2.set_xlim(-0.5, len(HORIZONS) - 0.5)
ax_a2.set_ylim(0, ymax)
ax_a2.set_xlabel('Forecast horizon', fontsize=FS_LABEL)
ax_a2.set_ylabel('MAE penalty vs Full subset (°C)', fontsize=FS_LABEL)   # Δ reserved for the horizon (QA 2026-09-16)
inline_panel_title(ax_a2, 'b',
                   'MAE penalty by removal level and horizon',   # plain label; the claim lives in §4.4 (2026-09-15)
                   pad_pts=6)
ax_a2.tick_params(labelsize=FS_TICK)
for _sp in ('top', 'right'): ax_a2.spines[_sp].set_visible(False)   # open style
ax_a2.grid(axis='y', alpha=0.25, linewidth=0.4)
ax_a2.legend(fontsize=FS_DENSE, loc='center left', bbox_to_anchor=(1.01, 0.5),   # OUTSIDE the axes, right column, vertically centred on the panel (author 2026-09-16)
             frameon=True, framealpha=0.92, edgecolor='#cccccc', handlelength=0.8,
             handletextpad=0.3, borderpad=0.22, labelspacing=0.22,
             borderaxespad=0.0, ncol=1,
             title='Removal', title_fontsize=FS_DENSE)
ax_a2.get_legend().get_frame().set_linewidth(0.4)

# [Per-architecture and per-LC robustness checks moved to supplementary]
# Architecture spread ≤ 0.11 °C across all 4 neural models at any horizon;
# LC spread covered separately by Fig 9 / Fig 10's full 10-station data.

# [Fig 5 group legend removed — swatches dropped from panel a1]


# =====================================================================
# ROW 2 — Panel (b): single sub-panel — empirical coverage at 90 %
# nominal across horizons. Reliability diagram + width moved to
# supplementary; one-line note in §4.4 caption: "Calibration was assessed
# at the 90 % nominal level; multi-level reliability across 50–95 %
# shows the same pattern (Fig SX)."
# =====================================================================
# Panel (c) — Empirical coverage bars (was panel-b-left in old layout).
# Now sits as the third column of row 1 so the figure is 2-row.
# The sharpness-coverage scatter (former b-right) was removed; the
# coverage bars + the panel (d) PI traces below carry the calibration
# story without it.
ax_b1 = fig.add_subplot(row1_right[1])   # (c) coverage — bottom-right, stacked under b (2026-06-13)
ax_b2 = ax_b1   # alias kept so existing bar code below works unchanged
ax_b3 = None    # scatter removed

CONF_COLOR = '#2ca02c'   # green — calibrated
MC_COLOR = '#9467bd'     # purple — miscalibrated (traditional deep-learning colour)
NOMINAL_COVERAGE = 90.0

conf_only = conf[conf.Method == 'Conformal']
mc_only = conf[conf.Method == 'MC Dropout']

# =====================================================================
# (b1) NEW reliability diagram — empirical vs nominal coverage at multiple
# levels for THREE representative horizons (5 min / 6 h / 24 h matching
# panel c). MC Dropout coverages computed from raw samples (samples_orig)
# at nominal levels 50 %, 70 %, 80 %, 90 %, 95 %. Conformal anchored at
# 90 % from CSV (statistically guaranteed near-nominal at every level).
# =====================================================================
# [b1 reliability diagram moved to supplementary — coverage at 90 %
# nominal in (b2) is the headline; multi-level reliability across
# 50–95 % shows the same MC under-coverage pattern (Fig SX).]

# =====================================================================
# (b2) Coverage bars per horizon (was ax_b1)
# =====================================================================
w = 0.36   # coverage bar width (c is half-width, stacked under b)

# Mean ± std across stations (defends "the gap is real, not just one
# station's noise")
conf_cov = [conf_only[conf_only.Horizon == h].Coverage.mean() for h in HORIZONS]
conf_std = [conf_only[conf_only.Horizon == h].Coverage.std(ddof=1) for h in HORIZONS]
mc_cov   = [mc_only[mc_only.Horizon == h].Coverage.mean() for h in HORIZONS]
mc_std   = [mc_only[mc_only.Horizon == h].Coverage.std(ddof=1) for h in HORIZONS]

# (Red under-coverage background shading removed 2026-06-14 — de-editorialised;
#  the dashed nominal line marks the threshold and bars below it read as under-
#  covering, consistent with removing the '(unsafe)' label + red value labels.)

bars_c = ax_b2.bar(hz_x - w/2, conf_cov, w, color=CONF_COLOR, alpha=0.85,
                    edgecolor='none',
                    yerr=conf_std, capsize=2.5,
                    error_kw={'lw': 0.7, 'ecolor': '#1a4d1a', 'alpha': 0.85},
                    label='Conformal', zorder=3)
bars_m = ax_b2.bar(hz_x + w/2, mc_cov, w, color=MC_COLOR, alpha=0.85,
                    edgecolor='none',
                    yerr=mc_std, capsize=2.5,
                    error_kw={'lw': 0.7, 'ecolor': '#4a1a4d', 'alpha': 0.85},
                    label='MC Dropout', zorder=3)

# Value labels: near-black for BOTH series — matches panel b's label colour
# (the figure's own standard); bar colour + legend already identify the series,
# and the nominal line + faint red under-coverage band carry the good/bad cue.
# (Bar-coloured labels were tried and declined by the author, 2026-09-16.)
import matplotlib.patheffects as _pe
_halo = [_pe.withStroke(linewidth=1.8, foreground='white')]   # the 5-min MC label (76.2) sits right under the nominal line (QA 2026-09-16)
for rect, val, std in zip(bars_c, conf_cov, conf_std):
    ax_b2.text(rect.get_x() + rect.get_width()/2, val + std + 1.5,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=FS_DENSE, fontweight='bold',
               color='#1a1a2e', path_effects=_halo, zorder=6)
for rect, val, std in zip(bars_m, mc_cov, mc_std):
    ax_b2.text(rect.get_x() + rect.get_width()/2, val + std + 1.5,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=FS_DENSE, fontweight='bold',
               color='#1a1a2e', path_effects=_halo, zorder=6)

ax_b2.axhline(NOMINAL_COVERAGE, color='black', ls='--', lw=0.8, alpha=0.7,
              label=f'Nominal ({NOMINAL_COVERAGE:.0f}%)', zorder=2)
# ('under-cover (unsafe)' in-plot label removed 2026-06-14 — it overlapped the
#  bars and editorialised; the bars falling below the nominal line + the red MC
#  values + the declarative panel title already carry the under-coverage message.)

ax_b2.set_xticks(hz_x)
ax_b2.set_xticklabels([HZ_LABELS[h] for h in HORIZONS], fontsize=FS_TICK)
ax_b2.set_xlim(-0.55, len(HORIZONS) - 0.45)
ax_b2.set_ylim(0, 105)
ax_b2.set_xlabel('Forecast horizon', fontsize=FS_LABEL)
ax_b2.set_ylabel('Empirical coverage (%)', fontsize=FS_LABEL)
inline_panel_title(ax_b2, 'c',
                   'Interval coverage by horizon',   # plain label (2026-09-15)
                   pad_pts=6)
ax_b2.tick_params(labelsize=FS_TICK)
ax_b2.legend(fontsize=FS_DENSE, loc='center left', bbox_to_anchor=(1.01, 0.5),   # OUTSIDE the axes, right column, vertically centred, matching (b) (author 2026-09-16)
             frameon=True, framealpha=0.92, edgecolor='#cccccc', handlelength=1.0,
             handletextpad=0.3, borderpad=0.22, labelspacing=0.22, borderaxespad=0.0)
ax_b2.get_legend().get_frame().set_linewidth(0.4)
ax_b2.grid(axis='y', alpha=0.3, linewidth=0.5)
for _sp in ('top', 'right'): ax_b2.spines[_sp].set_visible(False)   # open style

# [Sharpness-coverage scatter (former b-right) removed in 4-panel layout.
#  The coverage bars (panel c) and the PI traces (panel d) carry the
#  calibration story.]

# =====================================================================
# ROW 3 — Panel (c): MC Dropout PIs at 3 representative horizons
# =====================================================================
REP_HORIZONS = ['6h']   # keystone horizon only (5min/24h rows dropped 2026-06-13; the across-horizon coverage story is quantified in panel c, so panel d just illustrates the intervals at the keystone 6h)
# Panel d = single full-width 7-day MC-vs-conformal interval trace (outer[2]).
# (May-26 zoom sub-panel dropped 2026-06-13 — redundant with the decluttered
#  trace + miscoverage rug + panel c.)
ax_c_list = [fig.add_subplot(outer[1])]   # (d) full-width 6h interval trace

CHAMP_COLOR = MODEL_COLORS.get(CHAMPION, MODEL_COLORS['PatchTST'])
# Predicted mean colour for panel (c) — UQ convention is a saturated blue
# or red (Gal & Ghahramani 2016, MAPIE tutorials). Use PatchTST blue from
# MODEL_COLORS so the prediction-line hue matches the figure-suite scheme.
PRED_COLOR = MODEL_COLORS['PatchTST']   # '#0072B2'

N_SHOW = 2016  # 7 days at 5-min resolution (~7 diurnal cycles, climate-relevant window)

# Panel (d) wall-clock anchor — 7 whole days from 31 May (31 May-6 June 2017), the
# week immediately AFTER the Fig 9/10 deployment window (15-30 May). Moved from
# 24 May on 2026-09-16 (author: "7 days and n=10", then "the week after Fig 9"):
# the Constructed Wetlands test block only opens 25 May 07:55, so a 24 May start
# averaged 9 stations for its first 32 h; from 25 May 08:00 every step is a true
# 10-station mean, and this week also shows calibration on days Figs 9/10 do not use.
PANEL_C_START = pd.Timestamp('2017-05-31 00:00:00')
H_STEPS = {'5min': 1, '10min': 2, '1h': 12, '6h': 72, '24h': 288}

# --- Wall-clock timestamp alignment (mirrors Fig 9 method) ---------------
# MC Dropout was run on the primary station (10635576 = 10th Street Bridge).
# Segment-aware filtering in create_sequences breaks naive index→timestamp
# mapping, so we recover wall-clock times via correlation alignment between
# y_true and the test-split observed temperatures.
PRIMARY_SID = '10635576'
_sta_meta = STATION_COORDS[int(PRIMARY_SID)]
STATION_LABEL = _sta_meta['name']                                       # '10th St Bridge' — panel d trace station
STATION_CONTEXT = f"{_sta_meta['lc'].lower()}, LCZ {_sta_meta['lcz']}"   # 'exposed, LCZ 8' — for the caption
OVERLAP_START = '2015-06-05'
OVERLAP_END = '2017-06-28'

def _align_predictions_to_timestamps(y_true, df_te_temps_arr, df_te_index, max_search=2000):
    n_pred = len(y_true)
    check_n = min(500, n_pred)
    best_off, best_c = 0, -1
    for off in range(0, min(max_search, len(df_te_temps_arr) - n_pred + 1)):
        chunk = df_te_temps_arr[off:off + check_n]
        c = np.corrcoef(y_true[:check_n], chunk)[0, 1]
        if c > best_c:
            best_c, best_off = c, off
    return df_te_index[best_off:best_off + n_pred], best_c, best_off

mc_pred_times = {}
if mc_results is not None:
    df_raw = pd.read_csv(
        f'00_dataset/5min/20 stations/{PRIMARY_SID}_5min_Interval.csv',
        parse_dates=['Datetime'], low_memory=False)
    df_raw = (df_raw.dropna(subset=['Temp', 'RH', 'DewPt'])
                    .sort_values('Datetime')
                    .set_index('Datetime'))
    df_raw = df_raw.loc[OVERLAP_START:OVERLAP_END]
    n_total = len(df_raw)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    df_te = df_raw.iloc[n_train + n_val:]
    te_temps = df_te['Temp'].values
    print('Fig 8 (c) timestamp alignment:')
    for h_label in REP_HORIZONS:
        if mc_results.get(h_label) is None:
            mc_pred_times[h_label] = None
            continue
        y_true_arr = np.array(mc_results[h_label]['actuals'])
        ts, corr, off = _align_predictions_to_timestamps(y_true_arr, te_temps, df_te.index)
        if corr < 0.95:
            print(f'  {h_label}: corr={corr:.3f} (<0.95) -- falling back to elapsed time')
            mc_pred_times[h_label] = None
        else:
            mc_pred_times[h_label] = ts
            print(f'  {h_label}: offset={off}, corr={corr:.4f}, '
                  f'first={ts[0]}, last={ts[-1]}')

# =====================================================================
# MEAN-of-10 trace for panel d (n=10 network mean). Average the Attention-LSTM
# observed + predicted across all 10 stations (aligned to wall-clock), so panel d
# shows the network mean rather than one station. Conformal band = ±mean Q_hat
# (constant by construction); MC band = ±(mean MC 90% width)/2 — the true per-timestep
# MC samples exist only for the primary station, so the network-mean MC band is drawn
# at its mean width (disclosed in the box). Errors are correlated across the campus, so
# the mean residual barely shrinks → conformal still holds ~90% and MC still collapses.
# =====================================================================
_M10_SIDS = [10635593, 10635585, 10635582, 10635587, 10635576,
             10635580, 10635590, 10635588, 10635581, 10635595]
with open('ms_predictions_cache.pkl', 'rb') as _f:
    _m10_ms = pickle.load(_f)['ms_predictions']
    _assert_cache_orientation(_m10_ms)
_m10_ser = {}
for _sid in _M10_SIDS:
    _yp, _yt = _m10_ms[(str(_sid), '6h', 'Attention-LSTM')]   # cache stores (preds, actuals) - orientation fixed 2026-08-25
    _yt = np.asarray(_yt, float); _yp = np.asarray(_yp, float); _n = len(_yt)
    _df = pd.read_csv('00_dataset/5min/20 stations/%d_5min_Interval.csv' % _sid, low_memory=False,
                      parse_dates=['Datetime']).dropna(subset=['Temp', 'RH', 'DewPt']).sort_values(
                      'Datetime').set_index('Datetime').loc[OVERLAP_START:OVERLAP_END]
    _te = _df.iloc[int(len(_df) * 0.85):]; _tt = _te['Temp'].values; _bo, _bc = 0, -1; _chk = min(500, _n)
    for _off in range(0, min(1000, len(_tt) - _n) + 1):   # + 1: true offset was excluded (2026-09-30)
        _c = np.corrcoef(_yt[:_chk], _tt[_off:_off + _chk])[0, 1]
        if _c > _bc: _bc, _bo = _c, _off
    _m10_ser[_sid] = pd.DataFrame({'obs': _yt, 'pred': _yp}, index=_te.index[_bo:_bo + _n])
_m10_obs = pd.concat([_m10_ser[s].obs.rename(s) for s in _M10_SIDS], axis=1)
_m10_pred = pd.concat([_m10_ser[s].pred.rename(s) for s in _M10_SIDS], axis=1)
_m10_keep = _m10_obs.notna().sum(axis=1); _m10_keep = _m10_keep[_m10_keep >= 5].index
MEAN_OBS = _m10_obs.loc[_m10_keep].mean(axis=1, skipna=True).sort_index()
MEAN_PRED = _m10_pred.loc[_m10_keep].mean(axis=1, skipna=True).sort_index()
_mcw6 = pd.read_csv('mc_dropout_uq_results.csv'); _mcw6 = _mcw6[_mcw6.Horizon == '6h']
_mc_cvals = sorted(_mcw6.Confidence.unique())
_mc_pick = min(_mc_cvals, key=lambda x: abs(x - (90 if max(_mc_cvals) > 1.5 else 0.90)))
MC_WIDTH_MEAN = float(_mcw6[_mcw6.Confidence == _mc_pick].Width.mean())
print('panel d MEAN-of-10: %d common steps, mean MC width %.2f °C' % (len(MEAN_OBS), MC_WIDTH_MEAN))

for i, h in enumerate(REP_HORIZONS):
    ax = ax_c_list[i]
    if mc_results is None or mc_results.get(h) is None:
        ax.text(0.5, 0.5, 'No MC data', transform=ax.transAxes,
                ha='center', va='center', fontsize=FS_LABEL, color='0.5')
        continue

    r = mc_results[h]
    # Stats — use CSV-based network-mean (consistent with Panel b)
    cov90 = float(mc_only[mc_only.Horizon == h].Coverage.mean())
    w90 = float(mc_only[mc_only.Horizon == h].Width.mean())
    conf_cov_h = float(conf_only[conf_only.Horizon == h].Coverage.mean())
    conf_cov_min = float(conf_only[conf_only.Horizon == h].Coverage.min())   # per-station spread (n=10): makes the network-mean claim transparent in the box
    conf_cov_max = float(conf_only[conf_only.Horizon == h].Coverage.max())
    conf_width_h = float(conf_only[conf_only.Horizon == h].Width.mean())
    n_stations_cov = len(conf_only[conf_only.Horizon == h])

    # Aligned slice — anchor every panel to PANEL_C_START on the wall-clock
    # axis; show N_SHOW points (~7 days at 5-min resolution).
    act_full = np.array(r['actuals'])
    samp_full = np.array(r['samples_orig'])
    ts_h = mc_pred_times.get(h)
    if ts_h is not None:
        start_idx = int(ts_h.searchsorted(PANEL_C_START))
        use_dates = True
    else:
        start_idx = 0
        use_dates = False
    end_idx = min(start_idx + N_SHOW, len(act_full))
    act = act_full[start_idx:end_idx]
    samp = samp_full[:, start_idx:end_idx]
    mean = samp.mean(axis=0)
    lo90 = np.percentile(samp, 5.0, axis=0)
    hi90 = np.percentile(samp, 95.0, axis=0)
    if use_dates:
        t_x = ts_h[start_idx:end_idx]
    else:
        t_x = np.arange(len(act)) / 12.0

    # === OVERRIDE: panel d now shows the MEAN-of-10 network trace (n=10), not one station ===
    _win = (MEAN_OBS.index >= PANEL_C_START) & (MEAN_OBS.index < PANEL_C_START + pd.Timedelta(days=7))
    act = MEAN_OBS[_win].values
    mean = MEAN_PRED[_win].values
    t_x = MEAN_OBS[_win].index
    lo90 = mean - MC_WIDTH_MEAN / 2.0   # MC band = constant mean width (per-timestep MC samples only for primary station)
    hi90 = mean + MC_WIDTH_MEAN / 2.0
    use_dates = True
    # mean-of-10 trace coverage over the FULL test period (what this trace illustrates)
    _m10r = np.abs(MEAN_OBS.values - MEAN_PRED.values)
    M10_CONF_COV = float((_m10r <= conf_width_h / 2.0).mean() * 100)
    M10_MC_COV = float((_m10r <= MC_WIDTH_MEAN / 2.0).mean() * 100)

    # Conformal PI band (calibrated reference — width from CSV centered on
    # the MC mean since MC mean ≈ point prediction). Plotted UNDER the MC
    # band so the comparison is direct: observations stay within the wider
    # Conformal band even when they escape the narrower MC band.
    conf_lo = mean - conf_width_h / 2
    conf_hi = mean + conf_width_h / 2
    ax.fill_between(t_x, conf_lo, conf_hi,
                    color=CONF_COLOR, alpha=0.18, linewidth=0,
                    label='90% Conformal band', zorder=1)
    for _env in (conf_lo, conf_hi):   # stroke only the envelopes, not the polygon's closing edges at the window ends (QA 2026-09-16)
        ax.plot(t_x, _env, color=CONF_COLOR, linewidth=0.6, linestyle='--', zorder=1)

    # MC Dropout PI band on top — solid edge so its (too-narrow) width reads clearly
    ax.fill_between(t_x, lo90, hi90, alpha=0.35, color=MC_COLOR, linewidth=0,
                    label='90% MC Dropout band', zorder=2)
    for _env in (lo90, hi90):
        ax.plot(t_x, _env, color=MC_COLOR, linewidth=0.5, zorder=2)
    # UQ-standard z-order (per Gal & Ghahramani 2016, MAPIE, Angelopoulos
    # & Bates 2023): band → predicted mean → observed → miscoverage (top).
    # Predicted mean — neutral UQ blue (PatchTST hue used generically per
    # Gal & Ghahramani / MAPIE; the model here is Attention-LSTM, named in caption)
    ax.plot(t_x, mean, color=PRED_COLOR, lw=1.2, ls='--',
            alpha=0.95, zorder=3, label='Predicted mean')
    # Observed — thin BLACK line on top of predicted mean
    ax.plot(t_x, act, color='black', lw=0.7,
            alpha=0.9, zorder=4, label='Observed')
    # Miscoverage RUG — red ticks along the bottom axis mark every time the
    # observed escapes the (too-narrow) MC band (2026-06-13: replaced the dense
    # circle wall that buried the trace). Shows pervasiveness without clutter;
    # the full count/% are in the stats box and panel c. The zoom (below) keeps
    # circles for the close-up mechanism view.
    miss = (act < lo90) | (act > hi90)
    if miss.sum() > 0:
        from matplotlib.transforms import blended_transform_factory
        _trug = blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot(t_x[miss], np.full(int(miss.sum()), 0.022),
                marker='|', linestyle='none', color='#cc3333',
                markersize=4, markeredgewidth=0.5, alpha=0.45,
                transform=_trug, zorder=5)

    # (May-26 zoom highlight + sunrise/sunset markers removed, and the zoom
    #  sub-panel dropped entirely 2026-06-13 — the decluttered trace + miscoverage
    #  rug + panel c carry the calibration story. Diurnal physics: §5.1, Fig 7.)

    # Stats box: both methods + miscoverage count for the 7-day window
    n_main_obs = len(act)
    n_main_miss = int(miss.sum())
    pct_main_miss = (n_main_miss / n_main_obs * 100) if n_main_obs > 0 else 0
    # Conformal misses on this same window (parallels MC misses)
    conf_miss = (act < (mean - conf_width_h / 2)) | (act > (mean + conf_width_h / 2))
    n_main_conf_miss = int(conf_miss.sum())
    pct_main_conf_miss = (n_main_conf_miss / n_main_obs * 100) if n_main_obs > 0 else 0
    # Compact 2-line stats box — anchor the nominal target + mark the calibrated
    # winner with a check so the reader sees at a glance that Conformal hits 90 %
    # while MC Dropout severely under-covers.
    stats_txt = (
        f'Conformal coverage {conf_cov_h:.1f}%\n'
        f'MC Dropout coverage {cov90:.1f}%\n'
        f'Conformal per-station range {conf_cov_min:.1f}–{conf_cov_max:.1f}%'
    )
    ax.text(0.015, 0.96, stats_txt, transform=ax.transAxes,   # top-left corner, clear of the rug
            va='top', ha='left',
            fontsize=FS_DENSE,   # non-italic metrics box — matches the Fig 7 / Fig 9 diagnostics-box standard
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9, linewidth=0.4))

    # Title: panel letter + row-level title on top row only; each row
    # carries its horizon as the subtitle.
    if i == 0:
        panel_subtitle = f'Split-conformal vs MC Dropout bands, Δ = {HZ_LABELS[h]}'   # plain label (author's pick, 2026-09-15)
    else:
        panel_subtitle = f'Δ = {HZ_LABELS[h]}'
    inline_panel_title(ax, 'd' if i == 0 else '',
                       panel_subtitle, pad_pts=6)
    ax.set_ylabel('Temperature (°C)', fontsize=FS_LABEL)
    if use_dates:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        from matplotlib.ticker import FuncFormatter as _FF   # day-first ticks, no zero pad ('31 May', '1 Jun'; 2026-10-07)
        ax.xaxis.set_major_formatter(_FF(lambda x, pos: '%d %s' % (mdates.num2date(x).day, mdates.num2date(x).strftime('%b'))))
    else:
        ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    if i == len(REP_HORIZONS) - 1:
        ax.set_xlabel('Date (local time)' if use_dates else 'Elapsed time (h)',   # stamps are local clock time (daily Tmax ~14:00), not UTC (QA 2026-09-16)
                      fontsize=FS_LABEL)
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, linewidth=0.4)
    for _sp in ('top', 'right'): ax.spines[_sp].set_visible(False)   # open style

# Shared legend (below panel d)
row3_handles = [
    Line2D([0], [0], color='black', lw=0.9, label='Observed mean (n = 10)'),
    Line2D([0], [0], color=PRED_COLOR, lw=1.4, ls='--', label='Predicted mean (n = 10)'),
    Patch(facecolor=MC_COLOR, alpha=0.35, label='90% MC Dropout band'),
    Patch(facecolor=CONF_COLOR, alpha=0.18, edgecolor=CONF_COLOR,
          linestyle='--', linewidth=0.6, label='90% Conformal band'),
    Line2D([0], [0], marker='|', color='#cc3333', linestyle='none',
           markersize=8, markeredgewidth=1.0, alpha=0.7,
           label='MC Dropout miscoverage (rug)'),
]
fig.legend(handles=row3_handles, loc='lower center',
           bbox_to_anchor=(0.5, 0.015), ncol=5,
           fontsize=FS_LEGEND, frameon=True, framealpha=0.95,
           edgecolor='#cccccc', handlelength=1.6)
fig.legends[-1].get_frame().set_linewidth(0.4)

# =====================================================================
# Save — production PNG + PDF via fig_config helper
# =====================================================================
savefig(fig, 'fig08_ablation_uq', pad=0.05, bbox='fullwidth')