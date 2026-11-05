"""Fig 9 — Prediction Validation Bridge (leads to Fig 10 spatial application).

Layout (4 panels — exactly as described in §4.5 prose):
  (a) Top:    Observed vs Predicted hourly-mean heatmaps for 10 stations × 24 h
              (station names as LC-coloured y-ticks of Observed; dendrogram
              omitted — Figs 4 & 5 carry the taxonomy). Per-station MAE bars
              right of Predicted; best-model badges; horizontal colorbar
              below Observed.
  (b) Middle: Mean diurnal profiles for 5 land covers (Obs vs Pred + CI bands)
  (c) Bottom-left:  Obs vs Pred hourly-mean scatter (n=240), coloured by LC
  (d) Bottom-right: 2×2 quarter × LC error box plots (Morning/Afternoon/
                    Evening/Pre-dawn) — feeds directly into Fig 10's
                    quarter-resolved spatial UHI maps.

LC ordering across panels (b) and (d) matches Fig 10's column order:
  Built → Impervious → Exposed → Open green → Tree canopy
  (non-vegetated first, vegetated second — same as Fig 5 dendrogram leaves).

Narrative bridge:
  Figs 4-8 = model evaluation & mechanism discovery
  Fig 9    = "the champion actually predicts well" (demonstration)
  Fig 10   = spatial UHI application (uses the same predictions Fig 9 validates)
"""
import sys, io

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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_squared_error
import pickle

from fig_config import (FULL_W, MAX_H, DPI, MODEL_COLORS, MODEL_SHORT, CANDIDATES,
                         LC_COLORS, ORDERED_STATIONS_LCZ, SHORT_NAMES,
                         FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE, FS_DENSE_SM,
                         PANEL_LABEL_SIZE, apply_rcparams, savefig,
                         inline_panel_title,
                         LCZ_COLORS, VEG_COLORS, LCZ_SHORT, VEG_ORDER, DENDRO,
                         LW_ROOT, LW_BRANCH, LW_LEAF,
                         PAD_VEG, PAD_LCZ, PAD_LC,
                         COL_ROOT, COL_CONNECTOR,
                         BLW_VEG, BLW_LCZ, BLW_LC, HORIZONS, HZ_LABELS)
apply_rcparams()

CAND_COLORS = {k: MODEL_COLORS[k] for k in CANDIDATES}
ALL_MODELS = CANDIDATES + ['DLinear', 'Persistence']


# inline_panel_title() helper is now imported from fig_config (shared
# standard across all figures).

# =====================================================================
# Config
# =====================================================================
OVERLAP_START = '2015-06-05'
OVERLAP_END   = '2017-06-28'
HZ = '6h'
HZ_STEPS = 72
DATE_START = '2017-05-15'
DATE_END   = '2017-05-30'

n_sta = len(ORDERED_STATIONS_LCZ)
sids_ordered   = [s[0] for s in ORDERED_STATIONS_LCZ]
station_labels = [SHORT_NAMES.get(sn, sn) for _, sn, _, _, _ in ORDERED_STATIONS_LCZ]
# Abbreviated land-cover names for compact LCZ-LC y-axis subtitles.
# Built and Exposed already short; rest abbreviated.
LC_ABBREV = {
    'Built':       'Built',
    'Impervious':  'Imp.',
    'Exposed':     'Exp.',
    'Open green':  'Open',
    'Tree canopy': 'Tree',
}
lc_list  = [lc  for _, _, lc, _, _  in ORDERED_STATIONS_LCZ]
lcz_list = [lcz for _, _, _, lcz, _ in ORDERED_STATIONS_LCZ]
veg_list = [veg for _, _, _, _, veg in ORDERED_STATIONS_LCZ]

y_pos = np.arange(n_sta, dtype=float) * 1.0
y_max = y_pos[-1]

# =====================================================================
# Build dendrogram tree structure
# =====================================================================
full_tree = []
for veg in VEG_ORDER:
    veg_idx = [i for i in range(n_sta) if veg_list[i] == veg]
    if not veg_idx:
        continue
    lcz_groups = []
    seen_lcz = []
    for i in veg_idx:
        if lcz_list[i] not in seen_lcz:
            seen_lcz.append(lcz_list[i])
            lcz_idx = [j for j in veg_idx if lcz_list[j] == lcz_list[i]]
            lc_groups = []
            seen_lc = []
            for j in lcz_idx:
                if lc_list[j] not in seen_lc:
                    seen_lc.append(lc_list[j])
                    lc_idx = [k for k in lcz_idx if lc_list[k] == lc_list[j]]
                    lc_groups.append((lc_list[j], lc_idx))
            lcz_groups.append((lcz_list[i], lc_groups))
    full_tree.append((veg, lcz_groups))

# =====================================================================
# Load metrics (all models, all horizons, all stations)
# =====================================================================
metrics = pd.read_csv('multistation_perstation_metrics.csv')

# Best candidate model per station at 6h
best_model_per_station = []
for sid in sids_ordered:
    sub = metrics[(metrics.Station == sid) & (metrics.Horizon == HZ) &
                  (metrics.Model.isin(CANDIDATES))]
    best_model_per_station.append(sub.loc[sub['MAE'].idxmin(), 'Model'])

# mae_per_station is recomputed inside the per-station loop below using only
# the May 15-30 window so it is consistent with every other diagnostic in
# panel (a) (heatmaps, T_min/T_max markers, per-hour MAE box plot are all
# May 15-30 only). Using the full-test MAE here produced a 0.18°C mismatch
# between the bar legend (network mean = 1.54°C, full test) and the box
# plot legend (24-h mean = 1.73°C, window) — confusing for the reader.
mae_per_station = np.zeros(n_sta)

# =====================================================================
# Load predictions cache
# =====================================================================
cache_path = ('output/TF_for-upload_4(H100)__output__completed/'
              'output/run_2026-03-01_20-51/ms_predictions_cache.pkl')
with open(cache_path, 'rb') as f:
    preds_cache = pickle.load(f)['ms_predictions']
    _assert_cache_orientation(preds_cache)

# =====================================================================
# Build diurnal arrays: obs & pred per station (May 15-30 average)
# Also collect raw aligned pairs for Taylor + scatter
# =====================================================================
obs_by_hour  = np.zeros((n_sta, 24))
pred_by_hour = np.zeros((n_sta, 24))
# Cross-day std of OBSERVED and PREDICTED temperature per (station, hour).
# Captures inherent atmospheric variability (obs) and how well the model
# preserves that spread (pred). Both fed into the per-quarter "Mean
# variability" annotations under each heatmap.
obs_std_by_hour  = np.zeros((n_sta, 24))
pred_std_by_hour = np.zeros((n_sta, 24))
obs_tmin_hour = np.zeros(n_sta)
obs_tmax_hour = np.zeros(n_sta)
pred_tmin_hour = np.zeros(n_sta)
pred_tmax_hour = np.zeros(n_sta)
bias_per_station = np.zeros(n_sta)     # mean(pred - obs), signed
dt_err_per_station = np.zeros(n_sta)   # diurnal-amplitude error

# For quarter scatter (panel b): per-station means at each diurnal quarter
QUARTERS = [
    ('Morning',   6,  12, '06\u201312 h'),
    ('Afternoon', 12, 18, '12\u201318 h'),
    ('Evening',   18, 24, '18\u201324 h'),
    ('Pre-dawn',  0,   6, '00\u201306 h'),
]
quarter_obs  = {q: np.zeros(n_sta) for q, _, _, _ in QUARTERS}
quarter_pred = {q: np.zeros(n_sta) for q, _, _, _ in QUARTERS}

# For diurnal profiles: per-LC, 5-min resolution obs & pred + station names
N_BINS = 288
diurnal_obs_lc   = {lc: [] for lc in LC_COLORS}
diurnal_pred_lc  = {lc: [] for lc in LC_COLORS}
diurnal_names_lc = {lc: [] for lc in LC_COLORS}

print('Loading predictions and aligning timestamps...')
for i, sid in enumerate(sids_ordered):
    bm = best_model_per_station[i]
    lc = lc_list[i]

    # Load raw station data
    df = pd.read_csv(f'00_dataset/5min/20 stations/{sid}_5min_Interval.csv',
                     parse_dates=['Datetime'], low_memory=False)
    df = df.dropna(subset=['Temp', 'RH', 'DewPt']).sort_values('Datetime').set_index('Datetime')
    df = df.loc[OVERLAP_START:OVERLAP_END]

    n = len(df)
    t2 = int(n * 0.85)
    df_te = df.iloc[t2:]
    te_temps = df_te['Temp'].values

    # Find alignment offset using best model
    y_pred_bm, y_true_bm = preds_cache[(str(sid), HZ, bm)]   # cache stores (preds, actuals) - orientation fixed 2026-08-25
    n_pred = len(y_true_bm)

    best_offset = 0
    best_corr = -1
    check_n = min(500, n_pred)
    for test_offset in range(0, min(1000, len(te_temps) - n_pred) + 1):   # + 1: the true offset len - n was excluded (2026-09-30)
        chunk = te_temps[test_offset:test_offset + check_n]
        corr = np.corrcoef(y_true_bm[:check_n], chunk)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_offset = test_offset

    pred_times = df_te.index[best_offset:best_offset + n_pred]
    print(f'  {station_labels[i]:20s}: offset={best_offset}, corr={best_corr:.4f}')

    # Filter to May 15-30
    date_mask = (pred_times >= DATE_START) & (pred_times <= DATE_END + ' 23:59:59')
    date_idx = np.where(date_mask[:n_pred])[0]
    if len(date_idx) == 0:
        print(f'    WARNING: no data in May 15-30 for {station_labels[i]}')
        continue

    target_hours = pred_times[date_idx].hour.values
    yt_bm = y_true_bm[date_idx]
    yp_bm = y_pred_bm[date_idx]

    # Hourly means (best model) — used by the heatmaps in panel (a).
    for h in range(24):
        hmask = target_hours == h
        if hmask.any():
            obs_by_hour[i, h]  = yt_bm[hmask].mean()
            pred_by_hour[i, h] = yp_bm[hmask].mean()
            # Cross-day std at this (station, hour) — feeds per-quarter
            # "Mean variability" annotations on both heatmaps.
            if hmask.sum() > 1:
                obs_std_by_hour[i, h]  = float(yt_bm[hmask].std(ddof=1))
                pred_std_by_hour[i, h] = float(yp_bm[hmask].std(ddof=1))

    obs_tmin_hour[i] = np.argmin(obs_by_hour[i])
    obs_tmax_hour[i] = np.argmax(obs_by_hour[i])
    pred_tmin_hour[i] = np.argmin(pred_by_hour[i])
    pred_tmax_hour[i] = np.argmax(pred_by_hour[i])

    # Per-station MAE on the May 15-30 window (replaces the full-test value
    # so panel (a) is self-consistent to one window).
    mae_per_station[i] = float(np.mean(np.abs(yp_bm - yt_bm)))

    # Bias = mean(pred - obs) over all aligned 5-min pairs in window (signed).
    bias_per_station[i] = float(np.mean(yp_bm - yt_bm))

    # Diurnal-amplitude error = (predicted range) - (observed range).
    obs_amp  = obs_by_hour[i].max()  - obs_by_hour[i].min()
    pred_amp = pred_by_hour[i].max() - pred_by_hour[i].min()
    dt_err_per_station[i] = pred_amp - obs_amp

    # Quarter means for scatter panel (b)
    for q_name, h_start, h_end, _ in QUARTERS:
        hrs = list(range(h_start, h_end if h_end < 24 else 24))
        if h_end == 24:
            hrs = list(range(h_start, 24))
        qmask = np.isin(target_hours, hrs)
        if qmask.any():
            quarter_obs[q_name][i]  = yt_bm[qmask].mean()
            quarter_pred[q_name][i] = yp_bm[qmask].mean()

    # 5-min diurnal profiles for panel (d)
    target_minutes = pred_times[date_idx].hour.values * 60 + pred_times[date_idx].minute.values
    bins_5min = target_minutes // 5  # 0-287
    obs_5min = np.full(N_BINS, np.nan)
    pred_5min = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        bmask = bins_5min == b
        if bmask.any():
            obs_5min[b]  = yt_bm[bmask].mean()
            pred_5min[b] = yp_bm[bmask].mean()
    diurnal_obs_lc[lc].append(obs_5min)
    diurnal_pred_lc[lc].append(pred_5min)
    diurnal_names_lc[lc].append(station_labels[i])

del preds_cache

# Per-station T_min and T_max biases = predicted minus observed extremes
# (each station's diurnal-mean peak/trough). Network means show the
# asymmetric extreme-error pattern: small warm bias at the cool end,
# larger cold bias at the warm end (operationally important for
# heat-warning thresholds).
_obs_tmax_perstn   = obs_by_hour.max(axis=1)
_pred_tmax_perstn  = pred_by_hour.max(axis=1)
_tmax_bias_perstn  = _pred_tmax_perstn - _obs_tmax_perstn
_tmax_bias_network = float(_tmax_bias_perstn.mean())

_obs_tmin_perstn   = obs_by_hour.min(axis=1)
_pred_tmin_perstn  = pred_by_hour.min(axis=1)
_tmin_bias_perstn  = _pred_tmin_perstn - _obs_tmin_perstn
_tmin_bias_network = float(_tmin_bias_perstn.mean())

print('Data loaded. Building figure...')

# =====================================================================
# Quarter stats summary
# =====================================================================
print('Quarter validation (per-station May 15-30 means):')
for q_name, _, _, qlbl in QUARTERS:
    r2_q = r2_score(quarter_obs[q_name], quarter_pred[q_name])
    rmse_q = np.sqrt(mean_squared_error(quarter_obs[q_name], quarter_pred[q_name]))
    print(f'  {q_name:12s}: R2={r2_q:.4f}, RMSE={rmse_q:.2f}')

# =====================================================================
# FIGURE LAYOUT
# =====================================================================
# Station rows 2026-10-04 (author): the grey 'LCZ x - cover' subtitle under each station name is dropped, so a
# heatmap row holds one 6 pt line instead of two (row 0.258 -> 0.16 in). The heatmaps shorten from the TOP; every
# element below them keeps its distance in inches from the canvas bottom, the titles keep theirs from the top,
# and the canvas loses exactly the height the rows give up.
_H_OLD  = MAX_H * 0.9279   # trimmed 2026-09-04 (0.90); 0.87 on 2026-09-16: the c-row profiles are now SQUARE (1.10 x 1.10 in) and the saved height came off the figure
_RAT0, _HS0, _TOP0, _BOT0 = [3.48, 0.59, 0.22, 1.49, 0.47, 2.22], 0.27, 0.9544, 0.0785   # the 2026-09-17 grid (gs_main below)
_cell0  = (_TOP0 - _BOT0) * _H_OLD / (6 + 5 * _HS0)
_rows0  = [6 * _cell0 * r / sum(_RAT0) for r in _RAT0]   # row heights (in)
_sep0   = _HS0 * _cell0                                  # row separation (in)
_ROW_STA_IN = 0.16                                       # heatmap row per station (in); was _rows0[0] / n_sta = 0.258
_DROP   = _rows0[0] - _ROW_STA_IN * n_sta                # height the heatmap rows give up (in)
_rows1  = [_rows0[0] - _DROP] + _rows0[1:]
_H_NEW  = _H_OLD - _DROP
_FH     = _H_OLD / _H_NEW                                # figure-fraction lengths x _FH keep their inch size
_AXF    = _rows0[0] / _rows1[0]                          # heatmap axes-fraction offsets x _AXF keep their inch size
fig = plt.figure(figsize=(FULL_W, _H_NEW))

# Main grid: 4 rows now (added a thin metric-strip row directly below the
# heatmaps). Dendrogram dropped — Figs 4 & 5 already carry the taxonomy and
# the freed horizontal space lets the Obs/Pred heatmaps grow ~25% wider.
# top=0.94 — headroom for the panel-(a) label.
# right=0.95 — keeps vertical colorbar tick labels inside the figure.
gs_main = gridspec.GridSpec(6, 1,
                            height_ratios=_rows1,   # rows 2026-10-04: in inches; was [3.48, 0.59, 0.22, 1.49, 0.47, 2.22] (2026-09-17: row gap 0.27 with the colorbar label retired; top white trimmed; panel sizes kept)
                            hspace=_sep0 / (sum(_rows1) / 6), top=(_TOP0 * _H_OLD - _DROP) / _H_NEW, bottom=_BOT0 * _FH,   # rows 2026-10-04: same gaps and margins in inches; was hspace=0.27, top=0.9544, bottom=0.0785 (top 0.96->0.945: a/b titles touched the edge; bottom 0.045->0.0655 with +0.20 in so panel e's 45-degree cover names clear the canvas, 2026-09-16)
                            left=0.13, right=0.96)

# --- Row 0: Obs heatmap + Pred heatmap + (badge+MAE) column.
# Validation sub-panel was dropped — accuracy claims live in §4.5 prose,
# panel (b) per-LC profiles, and panel (c) aggregate scatter. Panel (a)
# now focuses purely on the "what does the prediction look like" visual.
COL_W = [4.2, 4.2, 0.95]   # Observed | Predicted | badges/MAE — best-model + MAE
                           # column on the RIGHT of panel b (Predicted); the two
                           # heatmaps stay edge-to-edge for the Obs-vs-Pred compare.
gs_top = gs_main[0].subgridspec(1, 3, width_ratios=COL_W, wspace=0.08)
ax_obs    = fig.add_subplot(gs_top[0, 0])
ax_pred   = fig.add_subplot(gs_top[0, 1])
ax_badges = fig.add_subplot(gs_top[0, 2])

# --- Row 1: under Observed only — legend (markers + model badges) on top,
# thin horizontal colorbar below (matched to Fig 10 thickness, 0.012 of
# figure height). Implemented with fig.add_axes() using the gridspec cell
# position so the legend axis takes the upper portion and the cbar takes
# a thin strip at the bottom.
gs_metric = gs_main[1].subgridspec(1, 3, width_ratios=COL_W, wspace=0.08)
_obs_cell_pos = gs_metric[0, 0].get_position(fig)   # Observed is the leftmost column
# Legend + colorbar are confined to the Observed-cell width so they read
# as a unit visually linked to the Observed heatmap above. The 6-entry
# legend wraps to 2 rows (ncol=3) below to fit cleanly in this narrower
# width without badge collisions. Legend top is dropped below row-1 top
# by `_bracket_buf` to leave a buffer for the two-line quarter bracket
# labels ("Pre-dawn / (00–06)", etc.) drawn below the heatmap axis
# above.
_cbar_h_fig       = 0.012 * _FH   # rows 2026-10-04: x _FH keeps the inch size
_gap              = 0.014 * _FH   # rows 2026-10-04: x _FH; gap between legend frame bottom and cbar top
_bracket_buf      = 0.020 * _FH   # rows 2026-10-04: x _FH; space at top of row 1 to clear the quarter
                             # bracket labels protruding from row 0 above
_cbar_drop        = 0.022 * _FH   # rows 2026-10-04: x _FH; ≈ 2× 9 pt font height — drops cbar + label
                             # away from the legend so the two elements
                             # read as separate visual layers
_legend_shift_down = 0.020 * _FH  # rows 2026-10-04: x _FH; ≈ 2× 9 pt — pushes the legend down toward
                             # the cbar so the empty space between them
                             # collapses (legend reads tighter without
                             # losing the cbar separation)
_cbar_y           = _obs_cell_pos.y0 - _cbar_drop
# Legend stays anchored to row-1 region but shifted DOWN by
# _legend_shift_down to compress the gap to the cbar.
_legend_y_lo      = _obs_cell_pos.y0 + _gap - _legend_shift_down
_legend_y_hi      = _obs_cell_pos.y1 - _bracket_buf - _legend_shift_down
ax_obs_legend = fig.add_axes([_obs_cell_pos.x0, _legend_y_lo,
                                _obs_cell_pos.width,
                                _legend_y_hi - _legend_y_lo])
ax_obs_legend.axis('off')
ax_cbar = fig.add_axes([_obs_cell_pos.x0, _cbar_y,
                         _obs_cell_pos.width, _cbar_h_fig])

# --- Row 2: SPACER — keeps the diurnal block (b) clear of the colorbar
#     above; nothing drawn here.
# --- Row 3: 5 diurnal profiles ---
_pos_mid = gs_main[3].get_position(fig)
gs_mid = gridspec.GridSpec(1, 5, figure=fig, wspace=0.25,
                           left=0.075, right=0.96,
                           top=_pos_mid.y1, bottom=_pos_mid.y0)   # rows c: own left margin, aligned to the station-label strip (author 2026-09-04)
ax_diurn = [fig.add_subplot(gs_mid[0, j]) for j in range(5)]

# --- Row 4: dedicated thin row for the shared Obs/Pred legend (placed
#     dynamically below using gs_main[4]). Avoids collision between the
#     centered legend and the centered "Hour of day" xlabel of the middle
#     diurnal panel.
# --- Row 5: scatter (left) + 2x2 box plots (right) ---
_pos_bot = gs_main[5].get_position(fig)
# Row d/e widths (2026-09-16, author: 'a big white gap on the left of panel e'): the scatter is a
# square as tall as the row, so its cell is sized to exactly that width; a fixed 0.55 in gutter
# holds panel e's y-label and ticks; everything else goes to the 2x2 box grid.
_sq_w_frac = (_pos_bot.y1 - _pos_bot.y0) * fig.get_figheight() / FULL_W   # square axes width as a figure fraction
_row_w     = 0.96 - 0.075
_gutter    = 0.80 / FULL_W   # 0.55 -> 0.80 in: room for panel e's 8 pt y-label and 7 pt ticks beside d's last x tick (author 2026-09-16)
_e_w_frac  = _row_w - _sq_w_frac - _gutter
gs_bot = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[_sq_w_frac, _e_w_frac],
                           wspace=_gutter / ((_row_w - _gutter) / 2.0),   # wspace is a fraction of the mean axes width
                           left=0.075, right=0.96,
                           top=_pos_bot.y1, bottom=_pos_bot.y0)   # rows d/e: own left margin, aligned to the station-label strip
ax_scat = fig.add_subplot(gs_bot[0, 0])
ax_scat.set_box_aspect(1)  # force square in figure coordinates
# Note: with set_box_aspect, the square axis is smaller than its rectangular
# cell. We use inline_panel_title(gs_cell=...) below to anchor the title to
# the CELL top (not the axis top) — this guarantees row-alignment with
# panel (d) titles regardless of where the square axis ends up inside its
# cell. No set_anchor needed.

gs_box22 = gs_bot[0, 1].subgridspec(2, 2, hspace=0.45, wspace=0.25)
ax_qbox = [fig.add_subplot(gs_box22[r, c]) for r in range(2) for c in range(2)]

# =====================================================================
# (a) Dendrogram intentionally REMOVED — Figs 4 & 5 carry the taxonomy.
#     Station names are rendered as LC-coloured y-tick labels of the
#     Observed heatmap (set after plot_heatmap call below). The (a) label
#     moves to ax_obs.
# =====================================================================

# =====================================================================
# Shared temperature colorbar range
# =====================================================================
all_temps = np.concatenate([obs_by_hour.ravel(), pred_by_hour.ravel()])
vmin_t = np.floor(all_temps.min())
vmax_t = np.ceil(all_temps.max())


# =====================================================================
# Helper: plot heatmap with discrete rows
# =====================================================================
def plot_heatmap(ax, data, title, show_markers=False, show_badge=False,
                 cmap='RdYlBu_r', vmin=None, vmax=None,
                 tmin_hr=None, tmax_hr=None):
    im = ax.pcolormesh(np.arange(25) - 0.5,
                       np.arange(n_sta + 1) - 0.5,
                       data,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='flat', edgecolors='white', linewidth=0.3)

    if show_markers and tmin_hr is not None and tmax_hr is not None:
        for i in range(n_sta):
            tmin_h = int(tmin_hr[i])
            tmax_h = int(tmax_hr[i])
            ax.plot(tmin_h, i, marker='o', color='white', markeredgecolor='black',
                    markeredgewidth=0.8, markersize=3.5, zorder=5)
            ax.plot(tmax_h, i, marker='*', color='black', markersize=5, zorder=5)

    if show_badge:
        for i in range(n_sta):
            bm = best_model_per_station[i]
            mc = CAND_COLORS[bm]
            ms = MODEL_SHORT[bm]
            ax.text(24.3, i, ms, fontsize=FS_DENSE_SM, va='center', ha='left',
                    color=mc, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.04', fc='white', ec=mc, lw=0.3))

    # Sunrise/sunset dotted lines removed — the black-dashed quarter
    # separators at 06/12/18 added below the plot_heatmap calls already
    # carry the morning/evening transition cue.

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(n_sta - 0.5, -0.5)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # Major ticks at hour BOUNDARIES (cell edges) so "06:00" aligns with
    # the dashed quarter separator at x=5.5 — convention is that "06:00"
    # marks the start of hour 6, not the middle of hour 6's cell.
    _major_hours = [0, 6, 12, 18]
    ax.set_xticks([h - 0.5 for h in _major_hours])
    ax.set_xticklabels([f'{h:02d}:00' for h in _major_hours],
                       fontsize=FS_TICK)
    # Minor ticks at every-3-hour boundaries (3, 9, 15, 21)
    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_minor_locator(FixedLocator([h - 0.5 for h in (3, 9, 15, 21)]))
    ax.tick_params(axis='x', which='major', length=2.5)
    ax.tick_params(axis='x', which='minor', length=1.2)
    ax.set_yticks(range(n_sta))
    ax.set_yticklabels([])
    # Title is set externally (2-line for first axis, 1-line for others)
    # so we skip the set_title call here.

    return im

# =====================================================================
# (a-left) Observed heatmap
# =====================================================================
im_obs = plot_heatmap(ax_obs, obs_by_hour,
                      'Observed (May 15\u201330 mean)',
                      show_markers=True,
                      tmin_hr=obs_tmin_hour, tmax_hr=obs_tmax_hour,
                      cmap='RdYlBu_r', vmin=vmin_t, vmax=vmax_t)

# Station labels: bold LC-coloured station name on the row centre (the grey
# 'LCZ x - cover' subtitle was dropped 2026-10-04: cover = name colour, keyed in
# panel d; LCZ in Fig. 2c and Table S14). Manual ax.text() as before.
ax_obs.set_yticks(range(n_sta))
ax_obs.set_yticklabels([])  # custom labels drawn below
# Left-side y-ticks visually connect each heatmap row to its station
# label in the LC-coloured group at left, mirroring the right-side ticks
# on the Predicted heatmap (which connect rows to the model badges).
ax_obs.tick_params(axis='y', which='major',
                   length=3.5, color='black', width=0.6, pad=3)
_yax_trans = ax_obs.get_yaxis_transform()
for i, (_sid, _sn, _lc, _lcz, _veg) in enumerate(ORDERED_STATIONS_LCZ):
    _name     = SHORT_NAMES.get(_sn, _sn)
    _lc_col   = LC_COLORS[_lc]
    # Station name — bold, LC-coloured, on the row centre (rows 2026-10-04)
    ax_obs.text(-0.025, i, _name,
                ha='right', va='center',
                fontsize=FS_DENSE_SM, fontweight='bold',
                color=_lc_col, transform=_yax_trans, clip_on=False)
# Panel (a) \u2014 single-line inline title on Observed; description-only on
# Predicted. pad_pts=18 lifts both above the top-mounted hour-tick labels.
inline_panel_title(ax_obs, 'a', 'Observed T, hourly mean (15\u201330 May)', pad_pts=18)
# Predicted heatmap is its own panel (b) \u2014 separated from Observed (a) per
# user request 2026-06-14; subsequent panels cascade c/d/e.
inline_panel_title(ax_pred, 'b', 'Predicted T+6 h, hourly mean',   # ', best model' dropped: it crowded the Best-model header and the badges say it (QA 2026-09-16)
                   pad_pts=18)

# =====================================================================
# (a-center) Predicted heatmap \u2014 same T_min/T_max markers but from the
# *predicted* diurnal so visual alignment with Observed is verifiable
# at a glance (any phase mismatch jumps out).
# =====================================================================
im_pred = plot_heatmap(ax_pred, pred_by_hour,
                       'Predicted T+6 h (May 15–30 mean, best model)',
                       show_markers=True,
                       tmin_hr=pred_tmin_hour, tmax_hr=pred_tmax_hour,
                       show_badge=False,
                       cmap='RdYlBu_r', vmin=vmin_t, vmax=vmax_t)

# Right-side y-ticks on the Predicted heatmap — small black ticks at each
# station row that lead the eye RIGHT to the model badge + MAE value in the
# adjacent ax_badges column. Acts like a tick axis, no labels, only lead lines.
ax_pred.set_yticks(range(n_sta))
ax_pred.set_yticklabels([])
ax_pred.yaxis.set_ticks_position('right')
ax_pred.tick_params(axis='y', which='major',
                    length=3.5, color='black', width=0.6)

# POLISH #2 — Overlay OBSERVED T_min / T_max positions on the Predicted
# heatmap. Stacked-marker design: a wide WHITE halo underneath provides
# contrast on any heatmap cell colour, with a dark outline on top giving
# the marker its shape. The result is clearly readable against both deep
# blue (T_min cells) and saturated red (T_max cells) without competing
# with the bold predicted marker.
_OBS_HALO   = 'white'
_OBS_RING   = '#222'
_OBS_HALO_W = 2.0
_OBS_RING_W = 1.0
for i in range(n_sta):
    _obs_min_h = int(obs_tmin_hour[i])
    _obs_max_h = int(obs_tmax_hour[i])
    _pred_min_h = int(pred_tmin_hour[i])
    _pred_max_h = int(pred_tmax_hour[i])
    if _obs_min_h != _pred_min_h:
        ax_pred.plot(_obs_min_h, i, marker='o', color='none',
                     markeredgecolor=_OBS_HALO, markeredgewidth=_OBS_HALO_W,
                     markersize=5.0, zorder=3.5)
        ax_pred.plot(_obs_min_h, i, marker='o', color='none',
                     markeredgecolor=_OBS_RING, markeredgewidth=_OBS_RING_W,
                     markersize=5.0, zorder=4)
    if _obs_max_h != _pred_max_h:
        ax_pred.plot(_obs_max_h, i, marker='*', color='none',
                     markeredgecolor=_OBS_HALO, markeredgewidth=_OBS_HALO_W,
                     markersize=7.0, zorder=3.5)
        ax_pred.plot(_obs_max_h, i, marker='*', color='none',
                     markeredgecolor=_OBS_RING, markeredgewidth=_OBS_RING_W,
                     markersize=7.0, zorder=4)

# ΔT bar removed — see gs_main comment above. Per-hour error is conveyed
# at the per-LC level by panel (b), at the aggregate level by panel (c),
# and decomposed by quarter × LC in panel (d).

# Quarter brackets below both heatmaps — square-bracket grouping that
# visually combines the hour-range with the quarter name (similar to a
# 'Shoots → SALT/HEAT/COMBINED' hierarchical grouping). Each bracket spans
# the quarter's hour range, with the quarter label centred below.
_Q_BRACKETS = [
    ('Pre-dawn\n(00–06 h)', 0,  6),    # x_start_hour, x_end_hour
    ('Morning\n(06–12 h)',  6, 12),
    ('Afternoon\n(12–18 h)', 12, 18),
    ('Evening\n(18–24 h)', 18, 24),
]
_Y_BRACK = -0.084 * _AXF   # rows 2026-10-04: x _AXF keeps the inch offset under the shorter heatmap; y of bracket's horizontal bar (axes fraction below axis)
_Y_TIP   = -0.064 * _AXF   # rows 2026-10-04: x _AXF; y of bracket's tiny upward tick at each end
_Y_LBL   = -0.104 * _AXF   # rows 2026-10-04: x _AXF; y of the quarter label (centred below bracket)
_Y_MAE   = -0.014 * _AXF   # rows 2026-10-04: x _AXF; y of 2-line "Mean MAE / 0.84" (Pred) and
                     # "Mean variability / 2.57" (Obs); va='top'.
                     # Whole annotation set shifted down ≈2 pt from the
                     # heatmap to add breathing room above.
_Y_N     = -0.174 * _AXF   # rows 2026-10-04: x _AXF; y of "n = X" annotation (below quarter label, Pred only)
# Per-quarter MAE + sample size for the Predicted-heatmap annotations.
# Each quarter pools |error| over (10 stations × 6 hours) = 60 hourly mean
# errors. Mean MAE shows the temporal-asymmetry of model accuracy
# directly under the heatmap (afternoon worst, evening/pre-dawn best).
_quarter_mae = {}
_quarter_n   = {}
for _qn, _h0, _h1, _ in QUARTERS:
    _hrs = list(range(_h0, min(_h1, 24)))
    _errs = [abs(pred_by_hour[_si, _h] - obs_by_hour[_si, _h])
             for _si in range(n_sta) for _h in _hrs]
    _quarter_mae[_qn] = float(np.mean(_errs))
    _quarter_n[_qn]   = len(_errs)

# Per-quarter MEAN VARIABILITY (cross-day σ averaged over stations and
# hours within the quarter) for OBSERVED and PREDICTED. Pure physical
# metric on Obs side; on Pred side it shows whether the model preserves
# the observed variability (under-prediction = amplitude attenuation).
_quarter_var      = {}
_quarter_var_pred = {}
for _qn, _h0, _h1, _ in QUARTERS:
    _hrs = list(range(_h0, min(_h1, 24)))
    _sig_obs  = [obs_std_by_hour[_si, _h]
                 for _si in range(n_sta) for _h in _hrs]
    _sig_pred = [pred_std_by_hour[_si, _h]
                 for _si in range(n_sta) for _h in _hrs]
    _quarter_var[_qn]      = float(np.mean(_sig_obs))
    _quarter_var_pred[_qn] = float(np.mean(_sig_pred))

for _ax in (ax_obs, ax_pred):
    _trans = _ax.get_xaxis_transform()
    for _lbl, _h0, _h1 in _Q_BRACKETS:
        # x-range in heatmap coords: cell centres at integer hours, edges at -0.5
        x0, x1 = _h0 - 0.5, _h1 - 0.5
        # Horizontal bar of the bracket
        _ax.plot([x0, x1], [_Y_BRACK, _Y_BRACK],
                 color='black', lw=0.6, clip_on=False,
                 transform=_trans, zorder=4)
        # Tiny upward ticks at each end (forms the └──┘ shape)
        _ax.plot([x0, x0], [_Y_BRACK, _Y_TIP],
                 color='black', lw=0.6, clip_on=False,
                 transform=_trans, zorder=4)
        _ax.plot([x1, x1], [_Y_BRACK, _Y_TIP],
                 color='black', lw=0.6, clip_on=False,
                 transform=_trans, zorder=4)
        # Quarter label centred below the bracket — regular weight per
        # project standard (region/category labels, not titles).
        _ax.text((x0 + x1) / 2, _Y_LBL, _lbl, transform=_trans,
                 ha='center', va='top',
                 fontsize=FS_DENSE_SM, color='black',
                 linespacing=1.0)
        # Per-quarter Mean MAE (above bracket) + sample size n (below
        # quarter label) — Predicted heatmap only. MAE is a comparison
        # metric so it has no meaning on the Observed heatmap alone.
        _qn = _lbl.split('\n')[0]
        if _ax is ax_pred:
            _ax.text((x0 + x1) / 2, _Y_MAE,
                     f'Profile MAE\n{_quarter_mae[_qn]:.2f} °C',   # 'Mean MAE' -> 'Profile MAE': MAE of the window-mean hourly profiles, not of the 5-min forecasts (QA 2026-09-16)
                     transform=_trans, ha='center', va='top',
                     fontsize=FS_DENSE_SM, color='black',
                     linespacing=1.0, clip_on=False)
            _ax.text((x0 + x1) / 2, _Y_N,
                     f'n = {_quarter_n[_qn]}',
                     transform=_trans, ha='center', va='top',
                     fontsize=FS_DENSE_SM, color='#555',
                     style='italic', clip_on=False)
        elif _ax is ax_obs:
            _ax.text((x0 + x1) / 2, _Y_MAE,
                     f'Day-to-day SD\n{_quarter_var[_qn]:.2f} °C',   # was 'Mean variability' + bare number; named and given its unit (author 2026-09-16: keep, it is the yardstick for the MAE strip under b)
                     transform=_trans, ha='center', va='top',
                     fontsize=FS_DENSE_SM, color='black',
                     linespacing=1.0, clip_on=False)
    # Dashed black quarter separators kept on the heatmap itself
    for _h in (6, 12, 18):
        _ax.axvline(_h - 0.5, color='black', linewidth=0.6,
                    linestyle='--', alpha=0.5, zorder=3)

# (Afternoon 12-18h brown dashed rectangle on the Predicted heatmap REMOVED 2026-09-16,
#  author's call: nothing in the caption or text referred to it, the "Mean MAE 0.86"
#  footer already marks the weak quarter, and the afternoon focus keeps one marker,
#  the red frame on panel e's Afternoon box, which hands over to Fig 10.)

# Combined badge + MAE column. Each row: model-short badge (PT/iT/LS/AL)
# next to the per-station MAE on the May 15-30 window. Replaces the old
# separate vertical-bar MAE column; saves width for the new validation
# sub-panel on the right.
_mae_arr = np.asarray(mae_per_station)
_mae_mean = float(_mae_arr.mean())
_mae_std  = float(_mae_arr.std(ddof=1))

ax_badges.set_xlim(0, 1)
ax_badges.set_ylim(n_sta - 0.5, -0.5)
ax_badges.axis('off')
# ONE header line over full-name rows 'LSTM (1.75 \u00b0C)' in regular text (author 2026-10-04; was two stacked column
# titles 'Best\nmodel' + 'MAE\n(\u00b0C)' over colour badges). The exact \u03bc/\u03c3 numbers live in \u00a74.5 prose.
_X_COL = -0.10   # 2026-10-04: shared left edge of the header and the rows (axes fraction; just right of panel b's row ticks)
_MODEL_NAME = {'Attention-LSTM': 'Attention-LSTM'}   # 2026-10-04: full model names in the column
# Header styled and placed exactly like the hour labels above panel b ('18:00'): same size, weight, colour and
# baseline, taken from the tick's own text transform (author 2026-10-04)
from matplotlib.transforms import blended_transform_factory as _btf_hdr
_tk = ax_pred.xaxis.get_major_ticks()[0]
_hdr_tr, _hdr_va, _ = ax_pred.get_xaxis_text2_transform(_tk.get_pad() + _tk.get_tick_padding())
ax_badges.text(_X_COL, 1.0, 'Best model (MAE)', transform=_btf_hdr(ax_badges.transAxes, _hdr_tr),
               fontsize=FS_TICK, va=_hdr_va, ha='left', color='black')
for i in range(n_sta):
    bm = best_model_per_station[i]
    mc = CAND_COLORS[bm]
    ms = MODEL_SHORT[bm]
    lc_col = LC_COLORS[lc_list[i]]
    # Best model + MAE as regular text, 'LSTM (1.75 °C)' (author 2026-10-04; was a colour badge + bold MAE)
    ax_badges.text(_X_COL, i, f'{_MODEL_NAME.get(bm, bm)} ({_mae_arr[i]:.2f} °C)',
                   fontsize=FS_DENSE_SM, va='center', ha='left', color='black')

# Network MAE summary moved from the badge column to the bottom-center
# of the Predicted heatmap so the headline accuracy + T_max bias number
# lives directly under the heatmap it summarizes (rather than in the
# narrow per-station MAE column on the side). x=11.5 = midpoint of the
# 24-hour axis; y in axes-fraction below the bracket annotations.
ax_pred.text(11.5, -0.235 * _AXF,   # rows 2026-10-04: inch offset kept under the shorter heatmap
             f'Network MAE = {_mae_mean:.2f} °C (5-min forecasts)\n'   # basis named: raw 5-min forecasts, unlike the profile MAE strip (QA 2026-09-16)
             f'T$_{{min}}$ bias = {("%+.2f" % _tmin_bias_network).replace("-", "−")} °C\n'   # three lines: MAE / T_min bias / T_max bias (author 2026-09-17)
             f'T$_{{max}}$ bias = {("%+.2f" % _tmax_bias_network).replace("-", "−")} °C',
             transform=ax_pred.get_xaxis_transform(),
             fontsize=FS_TICK, va='top', ha='center',   # summary box at the 7 pt annotation size (2026-09-16)
             color='black', fontweight='normal', clip_on=False,   # regular weight (author 2026-09-17)
             linespacing=1.2,
             bbox=dict(boxstyle='round,pad=0.30', fc='white',
                       ec='#cccccc', lw=0.4, alpha=0.9), zorder=6)   # same frame as the c/d stat boxes (author 2026-09-17)

# =====================================================================
# (a-bot) Per-hour MAE strip removed \u2014 was outside \u00a74.5's stated scope
# (heatmaps, diurnal profiles, scatter, quarter\u00d7LC boxplots) and risked
# dragging panel (a) into deployment-strategy territory that \u00a75.3 already
# handles cleanly via the pooled-vs-per-station comparison (Table 5).
# =====================================================================

# =====================================================================
# Colorbar \u2014 horizontal, beneath the Observed heatmap.
# =====================================================================
cbar1 = fig.colorbar(im_obs, cax=ax_cbar, orientation='horizontal')
cbar1.set_label('')   # label row retired 2026-09-17: the a/b titles name T; the unit sits at the bar's end
ax_cbar.text(1.012, 0.5, '\u00b0C', transform=ax_cbar.transAxes, ha='left', va='center', fontsize=FS_TICK)
cbar1.ax.tick_params(labelsize=FS_TICK, pad=1)
cbar1.ax.xaxis.set_major_locator(MultipleLocator(2))

# Validation sub-panel removed \u2014 T_min/T_max accuracy claims live in
# \u00a74.5 prose; panels (b) and (c) carry the aggregate fit story.

# Legend for the markers — placed between row 0 and row 1 (model-badge key retired 2026-10-04)
from matplotlib.legend_handler import HandlerBase
class _ObsPair: pass
class _ObsPairHandler(HandlerBase):
    # the two hollow markers of panel b side by side, set to the left of the handle box so that the star stays clear of its label
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        arts = []
        for frac, mk, ms in ((0.04, 'o', 4.5), (0.56, '*', 6.2)):
            a = Line2D([-xdescent + frac * width], [-ydescent + 0.5 * height], marker=mk, linestyle='none', markerfacecolor='none',
                       markeredgecolor=_OBS_RING, markeredgewidth=0.9, markersize=ms)
            a.set_transform(trans); arts.append(a)
        return arts
leg_markers = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='black', markeredgewidth=0.8, markersize=4,
           label='T$_{min}$ hour'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
           markersize=6, label='T$_{max}$ hour'),
    _ObsPair(),   # the hollow markers of panel b (the observed hour, where it differs from the predicted) had no key (figure check 2026-10-02)
]
leg_labels = ['T$_{min}$ hour', 'T$_{max}$ hour', 'observed, in b']
# (model-badge key 'LS = LSTM' etc. retired 2026-10-04: the best-model column now spells the model names)
ax_obs_legend.legend(handles=leg_markers, labels=leg_labels, handler_map={_ObsPair: _ObsPairHandler()},
                     loc='lower left', bbox_to_anchor=(0, 0, 1, 1),
                     mode='expand',
                     fontsize=FS_DENSE_SM,   # 6 pt by necessity: at 7 pt the three expanded columns overlap inside the Observed-cell width (checked 2026-09-16); dense-block allowance
                     ncol=3, framealpha=0.9, edgecolor='#cccccc',
                     handletextpad=0.3, borderpad=0.35,
                     columnspacing=0.9, labelspacing=0.30)
ax_obs_legend.get_legend().get_frame().set_linewidth(0.4)   # one frame style for every box and legend: #cccccc, 0.4 pt (author 2026-09-17)

# =====================================================================
# (b) STRONG SCATTER — hourly means × 10 stations = 240 points
# =====================================================================
from numpy.polynomial.polynomial import polyfit as np_polyfit

all_obs_flat  = []
all_pred_flat = []
all_lc_flat   = []
for i in range(n_sta):
    for h in range(24):
        all_obs_flat.append(obs_by_hour[i, h])
        all_pred_flat.append(pred_by_hour[i, h])
        all_lc_flat.append(lc_list[i])

all_obs_flat  = np.array(all_obs_flat)
all_pred_flat = np.array(all_pred_flat)

# 1:1 line
smin = np.floor(min(all_obs_flat.min(), all_pred_flat.min())) - 0.5
smax = np.ceil(max(all_obs_flat.max(), all_pred_flat.max())) + 0.5
ax_scat.plot([smin, smax], [smin, smax], '--', color='#999999', linewidth=0.8, zorder=1, label='1:1')

# ±1°C band
ax_scat.fill_between([smin, smax], [smin - 1, smax - 1], [smin + 1, smax + 1],
                     alpha=0.08, color='gray', zorder=0)   # band keyed in the caption, not the legend (QA 2026-09-16)

# Scatter by land cover
for lc in LC_COLORS:
    idx = [j for j in range(len(all_lc_flat)) if all_lc_flat[j] == lc]
    if idx:
        ax_scat.scatter(all_obs_flat[idx], all_pred_flat[idx],
                        c=LC_COLORS[lc], s=11, alpha=0.65,   # same dot style as panel e (author 2026-09-16)
                        edgecolors='none',
                        label=lc, zorder=3)

# Regression line
coeffs = np_polyfit(all_obs_flat, all_pred_flat, 1)
reg_x = np.array([smin, smax])
reg_y = coeffs[0] + coeffs[1] * reg_x
ax_scat.plot(reg_x, reg_y, '-', color='#cc3333', linewidth=0.9, alpha=0.7,
             label='Fit', zorder=2)   # keyed in the legend again (author 2026-09-16: 'it does not have the lines'); slope stays in the stats box

# Stats \u2014 kept to the four canonical aggregate-accuracy metrics so panel
# (c) stays focused on its narrative role (overall fit). The hot/cold
# bias asymmetry is already implicit in panel (b) (predicted curve
# droops at peak / overshoots at trough) and is quantified in \u00a74.5
# prose; no need to double-cover it here.
r2_global   = r2_score(all_obs_flat, all_pred_flat)
rmse_global = np.sqrt(mean_squared_error(all_obs_flat, all_pred_flat))
bias_global = np.mean(all_pred_flat - all_obs_flat)
ax_scat.text(0.05, 0.95,
             f'R\u00b2 = {r2_global:.3f}\n'   # 3 decimals like the c insets and the text (QA 2026-09-16)
             f'slope = {coeffs[1]:.3f}\n'
             f'RMSE = {rmse_global:.2f} \u00b0C\n'
             f'Bias = {bias_global:+.2f} \u00b0C\n'
             f'n = {len(all_obs_flat)}',
             # T_min / T_max bias intentionally NOT repeated here \u2014 they live in
             # the panel-b header strip (near the diurnal curves they describe).
             transform=ax_scat.transAxes, fontsize=FS_DENSE, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#cccccc',
                       alpha=0.9, lw=0.4))

ax_scat.set_xlim(smin, smax)
ax_scat.set_ylim(smin, smax)
ax_scat.xaxis.set_major_locator(MultipleLocator(2)); ax_scat.yaxis.set_major_locator(MultipleLocator(2))   # identical ticks on both axes (author 2026-09-16)
ax_scat.set_aspect('equal', adjustable='box')
for _sp in ('top', 'right'): ax_scat.spines[_sp].set_visible(False)   # open style (2026-09-16)
ax_scat.set_anchor('W')   # left-anchor the square so d's spine aligns with c above (author 2026-09-04)
ax_scat.set_xlabel('Observed T (\u00b0C)', fontsize=FS_LABEL)
ax_scat.set_ylabel('Predicted T (\u00b0C)', fontsize=FS_LABEL)
ax_scat.tick_params(labelsize=FS_TICK)
ax_scat.grid(True, alpha=0.15, linewidth=0.3)
_h, _l = ax_scat.get_legend_handles_labels()
_order = [_l.index('1:1'), _l.index('Fit')] + [i for i, n in enumerate(_l) if n not in ('1:1', 'Fit')]   # lines first, then the five covers
ax_scat.legend([_h[i] for i in _order], [_l[i] for i in _order],
               fontsize=FS_DENSE, loc='upper left', bbox_to_anchor=(-0.02, -0.20), ncol=4,   # OUTSIDE, two rows under the panel: 5 covers + 1:1 + fit no longer fit under the cloud (author 2026-09-16)
               framealpha=0.9, handletextpad=0.3, borderpad=0.3, markerscale=1.2, columnspacing=0.8, handlelength=1.2)
ax_scat.get_legend().get_frame().set_linewidth(0.4); ax_scat.get_legend().get_frame().set_edgecolor('#cccccc')
# Panel (c) — single-line inline title anchored to the GRIDSPEC CELL top
# (gs_bot[0, 0]) so it aligns with panel (d) titles in the same row even
# though set_box_aspect(1) shrinks the square scatter axis inside its cell.
inline_panel_title(ax_scat, 'd',
                   'Observed vs predicted',   # 'hourly means' lives in the caption (2026-09-16)
                   pad_pts=16, gs_cell=gs_bot[0, 0])   # raised to the e) title line (2026-09-16)

# =====================================================================
# (d) 4 SUB-PANEL BOX PLOTS — one per diurnal quarter
#     Each panel: 5 LC box plots with jittered station points
# =====================================================================
LC_ORDER_BOX = ['Built', 'Impervious', 'Exposed', 'Open green', 'Tree canopy']
LC_SHORT_BOX = {'Tree canopy': 'Tree', 'Open green': 'Open',   # 'Green' -> 'Open' to match panel a (QA 2026-09-16)
                'Impervious': 'Imp.', 'Built': 'Built', 'Exposed': 'Exp.'}
Q_BG = ['#fffff0', '#fff8e1', '#f0f0ff', '#f5f5f5']

# Collect absolute errors per quarter x LC
q_lc_errors = {}
for qi, (q_name, h_start, h_end, _) in enumerate(QUARTERS):
    hrs = list(range(h_start, min(h_end, 24)))
    for lc in LC_ORDER_BOX:
        key = (q_name, lc)
        q_lc_errors[key] = []
        lc_idx = [i for i in range(n_sta) if lc_list[i] == lc]
        for si in lc_idx:
            for h in hrs:
                q_lc_errors[key].append(abs(pred_by_hour[si, h] - obs_by_hour[si, h]))

# Per-quarter y-limits (set in the loop) let each subplot's boxes fill the panel
# (reference style); a shared dashed reference line at the network-median |error|
# (drawn in every subplot) keeps cross-quarter magnitude comparable — boxes above
# the line are worse than the typical error, restoring the at-a-glance read.
grand_med_err = float(np.median([e for v in q_lc_errors.values() for e in v]))
print(f'  Panel e reference line (network-median |error|, all quarters): {grand_med_err:.3f} C')

# Panel (d) letter is now part of the inline title set inside the per-quarter
# loop above (qi == 0 special-case calls _inline_title).

np.random.seed(42)
_e_top = max(e for _v in q_lc_errors.values() for e in _v)   # ONE y-scale for all four boxes (author 2026-09-16)
for qi, (q_name, h_start, h_end, time_str) in enumerate(QUARTERS):
    ax = ax_qbox[qi]
    # Quarter background tints removed — keeps panel (d) white/clean.

    bp_data = [q_lc_errors[(q_name, lc)] for lc in LC_ORDER_BOX]

    # Presentation swapped 2026-09-16 (author): the 12 station-hour values are the data and
    # carry the land-cover colour (as in panel d); the box is a neutral grey summary drawn
    # over them, median on top.
    # Box = thin dashed grey outline, no fill, UNDER the dots (author 2026-09-16); median thin black on top.
    bp = ax.boxplot(bp_data, positions=range(len(LC_ORDER_BOX)),
                    widths=0.55, patch_artist=True,
                    showfliers=False, zorder=2,
                    boxprops=dict(facecolor='none', edgecolor='#aaaaaa', linewidth=0.6, linestyle=':'),   # light grey DOTTED underlay (author 2026-09-17)
                    medianprops=dict(color='black', linewidth=1.0),
                    whiskerprops=dict(color='#aaaaaa', linewidth=0.6, linestyle=':'),
                    capprops=dict(color='#aaaaaa', linewidth=0.6))
    for _box in bp['boxes']:   # patch_artist boxes ignore the linestyle in boxprops; set it on the patch
        _box.set_linestyle((0, (1, 2))); _box.set_facecolor('none'); _box.set_edgecolor('#aaaaaa'); _box.set_linewidth(0.6)
    for _med in bp['medians']:
        _med.set_zorder(5)

    # Every station-hour value as a semi-transparent coloured dot (land-cover colour, as in panel d)
    for li, lc in enumerate(LC_ORDER_BOX):
        errs = q_lc_errors[(q_name, lc)]
        jitter = np.random.uniform(-0.16, 0.16, size=len(errs))
        ax.scatter([li + j for j in jitter], errs,
                   c=LC_COLORS[lc], s=11, alpha=0.65,
                   edgecolors='none', zorder=4)

    # Common reference line: network-median |error| across ALL quarters/LCs, so
    # cross-quarter magnitude stays comparable despite the per-quarter y-scales
    # (boxes above the dashed line = worse than the typical error).
    ax.axhline(grand_med_err, color='#444', ls='--', lw=0.8, alpha=0.6, zorder=1)
    if qi == 0:   # key the dashed reference once, in the Morning box, under its median-label row (author 2026-09-17)
        ax.legend(handles=[Line2D([0], [0], color='#444', ls='--', lw=0.8, alpha=0.6)],
                  labels=[f'Network median {grand_med_err:.2f} °C'],
                  loc='upper right', bbox_to_anchor=(1.0, 0.84), fontsize=FS_DENSE,
                  framealpha=0.9, edgecolor='#cccccc', handlelength=1.4, borderpad=0.3, handletextpad=0.4)
        ax.get_legend().get_frame().set_linewidth(0.4)

    # Shared y-limit (author 2026-09-16; was per-quarter): the same scale in all four boxes so
    # quarters compare directly, with a top band reserved for the aligned label row.
    q_top = _e_top
    ax.set_ylim(-0.03 * q_top, q_top * 1.32)

    # Median value labels — aligned in a clean row near the top (axes-fraction
    # y), centred over each box, so the numbers scan easily across land covers.
    _xax = ax.get_xaxis_transform()   # x = data coords, y = axes fraction
    for li, lc in enumerate(LC_ORDER_BOX):
        med = np.median(q_lc_errors[(q_name, lc)])
        ax.text(li, 0.91, f'{med:.2f}', transform=_xax,
                ha='center', va='center', fontsize=FS_DENSE,
                fontweight='normal', color='black')   # medians regular, 6 pt like the stat boxes (author 2026-09-17)

    # (Reference-line key kept out of the panel — an in-plot legend collided
    #  with the aligned median-label row; the dashed grey line is defined in the
    #  Fig 9 caption: "= network-median |error| across all quarters".)
    ax.tick_params(labelsize=FS_TICK)
    ax.yaxis.set_major_locator(MultipleLocator(1))   # 0/1/2/3 on the shared scale (2026-09-16)
    ax.grid(axis='both', alpha=0.3, linewidth=0.4, linestyle=':')   # dotted grey grid on x as well (author 2026-09-17)
    ax.set_axisbelow(True)
    # Map qi → (row, col) in gs_box22 so each sub-title anchors to its own
    # cell. Using inline_panel_title for ALL 4 sub-axes (with empty letter
    # for the non-first ones) ensures every title uses the SAME positioning
    # logic and they line up across the panel. n=12 = 2 stations × 6 hours
    # of hourly mean errors per box; constant across all 20 boxes (mirrors
    # panel (b)'s "(n=2)" convention so the reader can read sample size
    # straight from the title).
    _r, _c = qi // 2, qi % 2
    ax.set_title(f'{q_name} ({time_str}, n = 12)',   # centered sub-header (2026-09-16)
                 fontsize=FS_LABEL, fontweight='bold', color='black', loc='center', pad=4)
    if qi == 0:
        inline_panel_title(ax, 'e', 'Absolute error by quarter and land cover',
                           pad_pts=16, gs_cell=gs_bot[0, 1])

    # x-labels: only on the bottom row (qi=2,3); full cover names at 45 degrees (author 2026-09-16)
    if qi >= 2:
        ax.set_xticks(range(len(LC_ORDER_BOX)))
        ax.set_xticklabels(LC_ORDER_BOX, fontsize=FS_TICK,
                           rotation=45, ha='right', rotation_mode='anchor')
    else:
        ax.set_xticks(range(len(LC_ORDER_BOX)))
        ax.set_xticklabels([])

    # y-label and y tick labels on the left column only: the scale is shared (author 2026-09-16)
    if qi in (0, 2):
        ax.set_ylabel('|Error| (\u00b0C)', fontsize=FS_LABEL)
    else:
        ax.tick_params(axis='y', labelleft=False)

# Afternoon sub-panel (panel e) -- red thick border, Fig-7-panel-d style: the
# model's weakest regime / the binding constraint Fig 10 deploys against.
# (Red Afternoon frame REMOVED 2026-09-16, author's call; the quarter is named in the sub-header and Fig 10's caption says which quarter it maps.)
for _qax in ax_qbox:   # open style on all four boxes, as Figs 3/5/7/8
    for _sp in ('top', 'right'): _qax.spines[_sp].set_visible(False)

# Fig 10 annotation removed \u2014 narrative bridge already established in \u00a74.5
# prose ("feeds directly into Fig 10's quarter-resolved spatial UHI maps").

# =====================================================================
# (d) DIURNAL PROFILES x 5 LAND COVERS
# =====================================================================
LC_ORDER = ['Built', 'Impervious', 'Exposed', 'Open green', 'Tree canopy']
# Single prediction hue for panel (b) — one colour across all five sub-panels
# (land cover is named by the black panel title, not encoded in the line colour);
# matches Fig 8d's "predicted = blue dashed".
PRED_COLOR_B = MODEL_COLORS['PatchTST']   # '#0072B2'
hours_axis = np.arange(N_BINS) * 5 / 60  # 0 to ~24

# Shared y-limits
all_diurn_vals = []
for lc in LC_ORDER:
    for arr in diurnal_obs_lc[lc] + diurnal_pred_lc[lc]:
        all_diurn_vals.extend(arr[~np.isnan(arr)])
dy_min = np.floor(min(all_diurn_vals)) - 1.0   # was -0.5; the Open green band ran into the inset box (QA 2026-09-16)
dy_max = np.ceil(max(all_diurn_vals)) + 0.5

for j, lc in enumerate(LC_ORDER):
    ax = ax_diurn[j]

    obs_arrs = np.array(diurnal_obs_lc[lc])
    pred_arrs = np.array(diurnal_pred_lc[lc])

    obs_mean  = np.nanmean(obs_arrs, axis=0)
    pred_mean = np.nanmean(pred_arrs, axis=0)

    if obs_arrs.shape[0] > 1:
        obs_lo  = np.nanmin(obs_arrs, axis=0)
        obs_hi  = np.nanmax(obs_arrs, axis=0)
        pred_lo = np.nanmin(pred_arrs, axis=0)
        pred_hi = np.nanmax(pred_arrs, axis=0)
    else:
        obs_lo = obs_hi = obs_mean
        pred_lo = pred_hi = pred_mean

    lc_col = LC_COLORS[lc]

    # Observed: black solid + gray envelope
    ax.fill_between(hours_axis, obs_lo, obs_hi, alpha=0.12, color='black',
                    label='Obs. range' if j == 0 else None)
    ax.plot(hours_axis, obs_mean, '-', color='black', linewidth=1.0,
            label='Observed' if j == 0 else None)

    # Predicted: colored DASHED line + colored envelope. Dashed distinguishes
    # predicted from the black observed by line STYLE (not just colour) — reads
    # even in the low-contrast grey Impervious panel; matches Fig 8d's
    # "predicted = dashed" convention.
    ax.fill_between(hours_axis, pred_lo, pred_hi, alpha=0.15, color=PRED_COLOR_B,
                    label='Pred. range' if j == 0 else None)
    ax.plot(hours_axis, pred_mean, '--', color=PRED_COLOR_B, linewidth=1.1,
            label='Predicted' if j == 0 else None)

    # (sunrise/sunset vertical markers removed — they cluttered the diurnal profiles without
    #  adding info, and the 18:00 "sunset" line was mislabelled, late-May Atlanta sunset ~20:30.
    #  The afternoon under-forecast is visible in the curves themselves + quantified in panels d/e.)

    # Stats per LC
    valid = ~np.isnan(obs_mean) & ~np.isnan(pred_mean)
    if valid.sum() > 2:
        r2_lc = r2_score(obs_mean[valid], pred_mean[valid])
        rmse_lc = np.sqrt(mean_squared_error(obs_mean[valid], pred_mean[valid]))
        dtr_obs = np.nanmax(obs_mean) - np.nanmin(obs_mean)
        dtr_pred = np.nanmax(pred_mean) - np.nanmin(pred_mean)
        dtr_pct = dtr_pred / dtr_obs * 100 if dtr_obs > 0 else 0
        ax.text(0.97, 0.03,
                f'R\u00b2 = {r2_lc:.3f}\nRMSE = {rmse_lc:.2f} \u00b0C\n'
                f'\u0394T = {dtr_pct - 100:+.0f}%',   # signed excess of the predicted diurnal range over the observed one; 0 = perfect amplitude (author 2026-09-16)
                transform=ax.transAxes, fontsize=FS_DENSE_SM,
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.12', fc='white',
                          ec='#cccccc', alpha=0.9, lw=0.4))   # same frame as every other box (2026-09-17)

    # Formatting
    ax.set_xlim(0, 24)
    ax.set_ylim(dy_min, dy_max)
    ax.yaxis.set_major_locator(MultipleLocator(2))   # integer 2-degree ticks on the shorter square axis (2026-09-16)
    for _sp in ('top', 'right'): ax.spines[_sp].set_visible(False)   # open style, as Figs 3/5/7/8 (author 2026-06-11; applied here 2026-09-16)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(['0', '6', '12', '18', '24'], fontsize=FS_TICK)
    ax.xaxis.set_minor_locator(MultipleLocator(3))
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    n_stas_lc = len(diurnal_obs_lc[lc])
    ax.set_title(f'{lc} (n = {n_stas_lc})',   # centered sub-header (2026-09-16); the row carries ONE panel title
                 fontsize=FS_LABEL, fontweight='bold', color='black',
                 loc='center', pad=4)
    if j == 0:
        inline_panel_title(ax, 'c', 'Diurnal profiles by land cover',
                           pad_pts=16, gs_cell=gs_mid[0, :])
    if j == 2:
        ax.set_xlabel('Hour of day', fontsize=FS_LABEL)   # axis labels at FS_LABEL across c/d/e (2026-09-16)

    if j == 0:
        ax.set_ylabel('Temperature (\u00b0C)', fontsize=FS_LABEL)
    else:
        ax.tick_params(axis='y', labelleft=False)

# (Afternoon 12-18h brown dashed bracket REMOVED from the diurnal panels (panel c) -- it
#  cluttered the curves without adding info; the afternoon under-forecast is visible in the
#  curves themselves + quantified in panels d/e. Afternoon framing stays in panel e + Fig 10.)

# Panel (b) letter is now part of the inline title set inside the per-LC
# loop above (j == 0 special-case calls _inline_title).

# Shared Obs/Pred legend in the dedicated row 4 of gs_main, sitting
# between the diurnal panels (row 3) and the scatter+box panels (row 5).
# Centered vertically inside this dedicated row so the centered legend
# never collides with the centered "Hour of day" xlabel above or the
# panel (c)/(d) titles below.
_legend_cell = gs_main[4].get_position(fig)
_legend_y    = (_legend_cell.y0 + _legend_cell.y1) / 2
handles, labels = ax_diurn[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels,   # two lines at the bottom right of the profile row (author 2026-09-16)
           loc='lower right', bbox_to_anchor=(0.96, _legend_cell.y0 + 0.06 / fig.get_figheight()),
           ncol=2, fontsize=FS_LEGEND, framealpha=0.9,
           edgecolor='#cccccc', handletextpad=0.3, borderpad=0.25,
           columnspacing=1.0)
fig.legends[-1].get_frame().set_linewidth(0.4)

# =====================================================================
# Save — use exact figsize (7.48 × 9.45 = Elsevier double-column).
# The global apply_rcparams() sets 'savefig.bbox' = 'tight' which would
# expand to 7.90 × 9.57 due to overflow text; locally override to keep
# exactly the figsize bounds.
# =====================================================================
_saved_bbox = plt.rcParams['savefig.bbox']
plt.rcParams['savefig.bbox'] = 'standard'
try:
    fig.savefig('fig09.png', dpi=DPI, facecolor='white')
    fig.savefig('fig09.pdf', dpi=DPI, facecolor='white')
finally:
    plt.rcParams['savefig.bbox'] = _saved_bbox
print('\nSaved fig09.png and fig09.pdf')
plt.close()
