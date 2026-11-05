"""
Figure 7 -- Physical validation of the 6-h regime transition (3-row, 3-col)

Panels (5 SQUARE scatters in a 3-col × 2-row block + legend; (f) full-width):
  Row 0: (a) Δ=5min | (b) Δ=10min | (c) Δ=1h
  Row 1: (d) Δ=6h*  | (e) Δ=24h   | [scatter-panel legend]
  Row 2: (f) DL advantage vs DLinear (%) by hour of day, all 5 horizons

Each phase-space panel (a)-(e): 500 LC-colored points, ±2°C tolerance band,
y=x reference, regression fit line, multi-metric stats box. SQUARE
(set_aspect('equal')) — true-45° y=x + undistorted band, essential for a
persistence diagram. 3-col fits 5 squares across the double-column width
with ~zero wasted space; figure stays compact (~6.8" tall) and
proportionate to its content. Panel (d) at 6h highlighted with red
border — the regime transition. The raw-data persistence collapse
(r 0.998, 0.995, 0.962, 0.482, 0.817 on timestamp-matched pairs) is read from the stat boxes.

Panel (f): DL advantage over DLinear by hour of day at ALL five horizons
(shared y, 6 h red-bordered) — bimodal sunrise/sunset peaks unique to 6 h
= the diurnal-decoupling regime; the controls (flat at other horizons) are
in-figure, so no Supplementary companion is needed. The old full-width
r-vs-Δ "summary" panel was CUT 2026-05-15 (redundant with the scatters +
Fig 3 / Table 3). Data: hour_conditional_mae.csv.
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')

from fig_config import (
    FULL_W, MAX_H,
    LC_COLORS, HORIZONS, HZ_LABELS,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE, HSPACE_PANELS,
    inline_panel_title,
    apply_rcparams, savefig,
)
apply_rcparams()

# =====================================================================
# Data loading — raw 5-min station temperature
# =====================================================================
lc_map = {
    10635581: 'Tree canopy', 10635595: 'Tree canopy',
    10635588: 'Open green', 10635590: 'Open green',
    10635582: 'Impervious', 10635587: 'Impervious',
    10635585: 'Built', 10635593: 'Built',
    10635576: 'Exposed', 10635580: 'Exposed',
}
LC_ORDER = ['Built', 'Impervious', 'Exposed', 'Open green', 'Tree canopy']
station_ids = list(lc_map.keys())

# Horizon → step in 5-min units
HORIZON_STEPS = {'5min': 1, '10min': 2, '1h': 12, '6h': 72, '24h': 288}

# =====================================================================
# Build phase-space samples — subsample 100 pts per LC per horizon
# (≈500 total per panel) for crisp Fig-9c-style rendering
# =====================================================================
PTS_PER_LC = 100  # 5 LCs × 100 pts = 500 pts per phase-space panel
rng = np.random.default_rng(0)

# Step 1: load raw temperature time series per station
print('Loading raw temperature series...')
station_T = {}
for sid in station_ids:
    df = pd.read_csv(f'00_dataset/5min/20 stations/{sid}_5min_Interval.csv',
                     parse_dates=['Datetime'], low_memory=False)
    # Modeled window, as in the model pipeline (notebook cell 87): rows with Temp, RH and DewPt present, 5 June 2015 to
    # 28 June 2017. Lag pairs are matched by TIMESTAMP below, so gaps and between-season breaks never pair unrelated rows
    # (2026-09-30: the earlier version paired rows by position over each full raw file, which runs past June 2017).
    s = (df.dropna(subset=['Temp', 'RH', 'DewPt']).sort_values('Datetime')
           .set_index('Datetime').loc['2015-06-05':'2017-06-28']['Temp'])
    assert not s.index.duplicated().any(), sid
    station_T[sid] = s

# Step 2: build (t_now, t_lag, lc) samples per horizon
phase_samples = {}        # hz -> dict('t_now', 't_lag', 'lcs')
all_horizon_pairs = {}    # hz -> dict per LC -> (t_now, t_lag) for all-lag r computation
for hz in HORIZONS:
    step = HORIZON_STEPS[hz]
    by_lc = {lc: ([], []) for lc in LC_ORDER}
    for sid in station_ids:
        s = station_T[sid]
        lagged = s.reindex(s.index + pd.Timedelta(minutes=5 * step))   # T at t + delta, matched by timestamp
        ok = lagged.notna().values
        lc = lc_map[sid]
        by_lc[lc][0].append(s.values[ok])
        by_lc[lc][1].append(lagged.values[ok])
    # Concatenate per LC
    by_lc = {lc: (np.concatenate(a), np.concatenate(b))
             for lc, (a, b) in by_lc.items() if len(a)}
    all_horizon_pairs[hz] = by_lc

    # Stratified subsample for visual scatter
    t_now_list, t_lag_list, lc_list = [], [], []
    for lc in LC_ORDER:
        if lc not in by_lc:
            continue
        a, b = by_lc[lc]
        n = len(a)
        if n > PTS_PER_LC:
            sub = rng.choice(n, size=PTS_PER_LC, replace=False)
            t_now_list.append(a[sub]); t_lag_list.append(b[sub])
            lc_list.extend([lc] * PTS_PER_LC)
        else:
            t_now_list.append(a); t_lag_list.append(b)
            lc_list.extend([lc] * n)
    phase_samples[hz] = {
        't_now': np.concatenate(t_now_list),
        't_lag': np.concatenate(t_lag_list),
        'lc': np.array(lc_list),
    }

# =====================================================================
# Compute FULL-data r, slope, n per horizon (for stats boxes — unbiased)
# Only these three feed the rendered figure: r → stat box "r=",
# slope → stat box + regression line, n → stat box "n (data)=".
# (Per-LC r and model-prediction r were used by the now-cut summary
# panel — removed 2026-05-15 along with their pickle load and the
# per-station r computation, keeping render fast and the script lean.)
# =====================================================================
r_full_by_hz = {}     # hz -> r over all pairs across stations
slope_full_by_hz = {} # hz -> slope of linear fit over all pairs
n_full_by_hz = {}     # hz -> total pair count
for hz in HORIZONS:
    by_lc = all_horizon_pairs[hz]
    a_all = np.concatenate([by_lc[lc][0] for lc in LC_ORDER if lc in by_lc])
    b_all = np.concatenate([by_lc[lc][1] for lc in LC_ORDER if lc in by_lc])
    r_full_by_hz[hz] = float(np.corrcoef(a_all, b_all)[0, 1])
    slope_full_by_hz[hz] = float(np.polyfit(a_all, b_all, 1)[0])
    n_full_by_hz[hz] = len(a_all)

# =====================================================================
# Figure layout — 3-row, 3-col: rows 0–1 = the five SQUARE phase-space
# scatters (a)–(e) + scatter legend in (1,2); row 2 = the full-width
# hour-conditional grid (f). 3-col fits 5 square scatters across the
# double-column width with ~zero wasted space and keeps the figure
# compact (~6.8" tall) — proportionate to the content, vs the 2-col
# variant which inflated to a near-full page for the same information.
# Scatters keep set_aspect('equal') — true-45° y = x + undistorted
# ±2 °C band, essential for a persistence diagram. The old r-vs-Δ
# summary panel was CUT (redundant with the scatters + Fig 3/Table 3).
# =====================================================================
# gaps 2026-10-04: rows converted to inches (same sizes as the old 4.60-in grid); only the legend -> (b) gap shrinks
_H0 = 4.60                                          # gaps 2026-10-04: old canvas height
_sum_h = (0.905 - 0.075) * _H0 / (1 + 2 * HSPACE_PANELS / 3)   # gaps 2026-10-04: old summed cell height (in)
_rows_in = [r * _sum_h / 3.22 for r in (1.30, 0.62, 1.30)]     # gaps 2026-10-04: old cell heights (in)
_sep_in = HSPACE_PANELS * _sum_h / 3                # gaps 2026-10-04: old row separation (in)
_ROW_GAP_CUT = 0.025                                # gaps 2026-10-04: legend -> (b) title band 0.145 -> 0.12 in
_H = _H0 - _ROW_GAP_CUT                             # gaps 2026-10-04: new canvas height
fig = plt.figure(figsize=(FULL_W, _H))  # 5x2 layout 2026-09-04: scatters a-e one row, (f) row below; gaps 2026-10-04: 4.60 -> _H
# Two empty spacer rows (rows 1 and 3) give DIFFERENTIAL row gaps under
# a single GridSpec hspace: row 1 spacer pushes (d/e)+legend down a touch
# below the (a/b/c) row; row 3 spacer pushes (f) down below (d/e)+legend.
# Cleaner than nested GridSpec and gives explicit control of two gaps.
outer = gridspec.GridSpec(5, 5, figure=fig,   # gaps 2026-10-04: 3 rows -> 5 (explicit spacer rows 1 and 3)
                           wspace=0.14, hspace=0,   # 5x2: tight columns, sharey-look scatters; gaps 2026-10-04: spacing now in the spacer rows
                           height_ratios=[_rows_in[0], _sep_in, _rows_in[1], _sep_in - _ROW_GAP_CUT, _rows_in[2]],   # gaps 2026-10-04: scatter | sep | legend row | sep - cut | (b) squares, in inches
                           left=0.065, right=0.985, top=1 - (1 - 0.905) * _H0 / _H, bottom=0.075 * _H0 / _H)   # gaps 2026-10-04: same top/bottom margins in inches

# Row 0: (a) 5min | (b) 10min | (c) 1h
# Row 1: [spacer]
# Row 2: (d) 6h*  | (e) 24h   | [scatter legend in (2,2)]
# Row 3: [spacer]
# Row 4: (f) hour-conditional grid, all 5 horizons (full width)
panel_horizons = ['5min', '10min', '1h', '6h', '24h']
panel_labels = ['a', 'b', 'c', 'd', 'e']
panel_positions = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

ax_panels = [fig.add_subplot(outer[r, c]) for r, c in panel_positions]

# =====================================================================
# Phase-space panels (a)–(e) — Fig-9c visual grammar
# =====================================================================
TOL = 2.0           # ±2°C tolerance band
xy_lo, xy_hi = 14, 38

for i, hz in enumerate(panel_horizons):
    ax = ax_panels[i]
    s = phase_samples[hz]
    is_6h = (hz == '6h')

    # ±2°C tolerance band around y=x
    xx = np.linspace(xy_lo, xy_hi, 100)
    ax.fill_between(xx, xx - TOL, xx + TOL,
                    color='#cccccc', alpha=0.30, zorder=1, linewidth=0)

    # y = x reference
    ax.plot([xy_lo, xy_hi], [xy_lo, xy_hi], '--',
            color='black', lw=0.6, alpha=0.55, zorder=2)

    # Scatter: filled circles with dark edge — Fig 9c grammar
    point_colors = [LC_COLORS[lc] for lc in s['lc']]
    ax.scatter(s['t_now'], s['t_lag'],
               c=point_colors, s=11, alpha=0.65,   # same dot style as Fig 9 d/e (author 2026-09-17)
               edgecolor='none', zorder=4)

    # Linear fit — slope from FULL data, intercept recomputed from FULL data
    # too so the line passes through the full-data centroid (not the subsample's).
    slope = slope_full_by_hz[hz]
    by_lc = all_horizon_pairs[hz]
    a_all = np.concatenate([by_lc[lc][0] for lc in LC_ORDER if lc in by_lc])
    b_all = np.concatenate([by_lc[lc][1] for lc in LC_ORDER if lc in by_lc])
    inter = float(np.mean(b_all) - slope * np.mean(a_all))
    fit_x = np.linspace(xy_lo, xy_hi, 50)
    ax.plot(fit_x, slope * fit_x + inter,
            color='#c0392b', lw=1.2, alpha=0.95, zorder=5)

    # Stats box (top-left) — sans-serif compact, matches the house
    # annotation style (cf. Fig 9 R²/RMSE blocks). All four metrics kept;
    # the persistence framing is carried by the panel title + caption, so
    # the labels can be plain (no monospace, no heavy subscripts).
    r_val = r_full_by_hz[hz]
    n_total = n_full_by_hz[hz]
    n_plot = len(s['t_now'])
    stats_txt = (f'r = {r_val:.3f}\n'
                 f'slope = {slope:.3f}')   # n(plot)/n(data) live in the caption (5x2 slim, 2026-09-04)
    ax.text(0.96, 0.04, stats_txt, transform=ax.transAxes,
            va='bottom', ha='right', zorder=10,   # bottom-right; above scatter (zorder 4) so dots don't draw over the card
            fontsize=FS_DENSE, linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white',
                      edgecolor='#cccccc', linewidth=0.4,   # frame style shared by every stat box and legend (2026-09-17)
                      alpha=0.9))   # translucent; uniform edge (red 6h accent removed 2026-09-04)

    # Title — "Persistence @ Δ = X" makes the model-free nature explicit.
    # Use inline_panel_title with gs_cell mode so titles align above the
    # equal-aspect axes (which may be smaller than the gridspec cell).
    if i == 0:   # ONE panel title over the five scatters (project standard, 2026-09-15); the Δ labels are sub-headers
        inline_panel_title(ax, 'a', 'Persistence: T at t + Δ versus T at t',
                           pad_pts=10.4, gs_cell=outer[0, :])   # gaps 2026-10-04: pad 14 -> 10.4 pt, title lift 0.13 -> 0.08 in
    ax.set_title(f'Δ = {HZ_LABELS[hz]}', fontsize=FS_LABEL, fontweight='bold',
                 pad=3, loc='center', color='black')   # centered sub-header, matching the (b) row

    # Highlight 6h panel with a full red border (regime transition; explained in caption).
    # In-panel "REGIME TRANSITION" badge removed 2026-06-12 — redundant with the red border
    # + red stats box + red fit line, and it crowded the corner annotations.
    # red 6h border removed (author 2026-09-04); uniform open style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(xy_lo, xy_hi); ax.set_ylim(xy_lo, xy_hi)
    ax.set_xticks([15, 20, 25, 30, 35]); ax.set_yticks([15, 20, 25, 30, 35])   # matching tick ladders (author 2026-09-04)
    ax.set_aspect('equal')   # square: true-45° y=x + undistorted band
    ax.set_anchor('S')       # bottom-anchor: square fills cell width, aligns with (f) below
    # Axis-label dedup: xlabel only on the bottom-row scatters ((d)=3, (e)=4),
    # so the whole top row (a,b,c) is uniformly x-label-free — the shared x-axis
    # is labelled once at the bottom (c's label was dropped 2026-06-12: it floated
    # mid-figure above the legend and was redundant). ylabel only on left-column
    # panels ((a)=0, (d)=3). Tick numbers stay on every panel (gapped layout).
    if i == 2:
        ax.set_xlabel('Observed T at time $t$  (°C)', fontsize=FS_LABEL)
    if i == 0:
        ax.set_ylabel('T at $t+\\Delta$  (°C)', fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    if i > 0:
        ax.tick_params(labelleft=False)   # y numbers only on (a); columns sit tight
    ax.grid(alpha=0.3, linewidth=0.5)

# =====================================================================
# (Old full-width r-vs-Δ "summary" panel CUT 2026-05-15. Its
# persistence-collapse content duplicated the (a)–(e) scatters; its
# model-vs-persistence content duplicated Fig 3 / Table 3; only its
# r-framing was unique and it was not load-bearing. §4.4's
# "Persistence-to-model comparison" prose was re-anchored on the
# scatter-panel r + Fig 3 skill. The hour-conditional 6 h mechanism
# panel below is now panel (f).)
# =====================================================================

# =====================================================================
# (Figure-level "Persistence forecast: T̂(t+Δ)=T(t)" banner removed —
# canvas-level subtitles are off-house-style; the persistence definition
# now lives in the Fig 7 caption instead. Panel titles + axis labels
# already carry the "raw observations, no model" framing.)
# =====================================================================

# =====================================================================
# Scatter-panel legend — describes the (a)-(e) phase-space panels only.
# Placed in the empty (1, 2) cell of the grid.
# =====================================================================
# 5x2 layout: no blank cell; the scatter legend moves to a horizontal strip below (f).
legend_handles = [
    Line2D([0], [0], color='#c0392b', lw=1.4,
           label='Persistence regression'),
    Line2D([0], [0], color='black', ls='--', lw=0.7,
           label='y = x (perfect persistence)'),
    Patch(facecolor='#cccccc', alpha=0.40,
          label=f'±{int(TOL)} °C around y = x'),   # was '±2°C tolerance band': "tolerance" is the 0.21 °C sensor accuracy in the text (figure check 2026-10-02)
] + [Line2D([0], [0], marker='o', linestyle='none',
            markerfacecolor=LC_COLORS[lc], markeredgecolor='none',
            markersize=5.5, alpha=0.85, label=lc)
     for lc in LC_ORDER]

_leg_cell = outer[2, :].get_position(fig)   # the legend row between the scatter row and (b) (row 2 since gaps 2026-10-04): the legend belongs to panel a (author 2026-09-17)
_leg7 = fig.legend(handles=legend_handles, loc='center',
           bbox_to_anchor=(0.5, (_leg_cell.y0 + _leg_cell.y1) / 2), ncol=4,
           fontsize=FS_LEGEND, frameon=True, framealpha=0.9,
           edgecolor='#cccccc', handlelength=1.4, columnspacing=1.0)
_leg7.get_frame().set_linewidth(0.4)

# =====================================================================
# Panel (f) — Hour-conditional DL advantage over DLinear at ALL FIVE
# horizons (Row 2, full-width 1×5 grid, shared y). The (a)–(e) scatters
# show the raw-data persistence collapse at 6 h; (f) shows WHEN in the
# day the DL advantage concentrates AND that the bimodal sunrise
# (06–09 h) / sunset (18–22 h) peaks are UNIQUE to the 6 h regime-
# transition horizon (absent at 5 min/10 min/1 h/24 h) — specificity is
# the mechanistic argument. Gets a full generous row now that the
# redundant r-vs-Δ summary panel was cut (which is what made the old
# 5-grid cramped). Self-contained: no Supplementary control needed.
# =====================================================================
DL_MODELS_G = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']
# Hour = TARGET wall-clock hour of each forecast. File regenerated 2026-09-22 with the NaN-aware correlation alignment
# (fig_script/_lbl_hour_regen.py copy in the scratchpad); the previous file carried station-specific offsets of 0-8 h.
hourly_df = pd.read_csv('hour_conditional_mae.csv')
# (f) panels live in the SAME outer columns as the scatters -> exact vertical alignment

_hours = np.arange(24)
gap_by_hz = {}
for _hz in HORIZONS:
    _g = []
    for _h in _hours:
        _dl = hourly_df[(hourly_df['Horizon'] == _hz) &
                        (hourly_df['Hour'] == _h) &
                        (hourly_df['Model'].isin(DL_MODELS_G))
                        ].groupby('Station')['MAE'].mean().mean()
        _dlin = hourly_df[(hourly_df['Horizon'] == _hz) &
                          (hourly_df['Hour'] == _h) &
                          (hourly_df['Model'] == 'DLinear')
                          ].groupby('Station')['MAE'].mean().mean()
        _g.append((1 - _dl / _dlin) * 100 if _dlin and _dlin > 0 else 0)
    gap_by_hz[_hz] = np.array(_g)
_all_g = np.concatenate(list(gap_by_hz.values()))
_ylo = float(np.floor(_all_g.min() / 5.0) * 5.0)
_yhi = float(np.ceil(_all_g.max() / 5.0) * 5.0)
_ypad = max(2.0, 0.08 * (_yhi - _ylo))

ax_f = []
for _i in range(5):
    _a = (fig.add_subplot(outer[4, _i]) if _i == 0   # gaps 2026-10-04: (b) row is now row 4
          else fig.add_subplot(outer[4, _i], sharey=ax_f[0]))
    ax_f.append(_a)

# (f)'s diverging bar colors are deliberately NOT green/red — those
# hexes (#2ca02c, #d62728) are Tree-canopy / Built in fig_config.LC_COLORS
# and used by the (a)–(e) scatter legend in THIS same figure. Reusing
# them for win/loss would semantically clash (one colour = two meanings).
# Dark blue / light blue (author's theme, 2026-09-17): a neutral pair that echoes no model colour
# (the old blue/amber were the PatchTST / LSTM hues of Figs 3 and 8).
_C_DL_WINS = '#2b4c8c'      # dark blue — DL beats DLinear (positive gap)
_C_DLIN_WINS = '#c3cfe0'    # light blue — DLinear beats DL (negative gap)
for _i, _hz in enumerate(HORIZONS):
    _a = ax_f[_i]
    _gap = gap_by_hz[_hz]
    _cols = [_C_DL_WINS if g > 0 else _C_DLIN_WINS for g in _gap]
    _a.bar(_hours, _gap, color=_cols, alpha=1.0, width=0.86,
           edgecolor='none', zorder=2)
    _a.axhline(0, color='black', linewidth=0.4, zorder=3)
    # day/night shading removed per user (cleaner panel)
    _is6 = (_hz == '6h')
    _a.set_title(HZ_LABELS[_hz], fontsize=FS_LABEL, fontweight='bold',
                 pad=3, loc='center', color='black')
    _a.set_xlim(-0.5, 23.5)
    _a.set_ylim(_ylo - _ypad, _yhi + _ypad)
    _a.set_box_aspect(1.0)   # square (f) panels (author 2026-09-04)
    _a.set_anchor('N')       # top-anchor: flush under the scatter columns
    _a.set_xticks([0, 6, 12, 18])
    _a.set_xticklabels(['0', '6', '12', '18'], fontsize=FS_TICK)   # standard tick size (match scatters)
    _a.tick_params(axis='y', labelsize=FS_TICK, labelleft=(_i == 0))   # standard tick size (match scatters)
    _a.tick_params(axis='x', length=2)
    if _i == 2:
        _a.set_xlabel('Target hour of day', fontsize=FS_LABEL)   # once, under the middle panel (author 2026-09-17); TARGET hour = the hour the forecast is valid for (2026-09-22)
    _a.grid(True, axis='y', alpha=0.2, linewidth=0.4)   # value-axis grid — matches Fig 3 g/h bar panels
    _a.set_axisbelow(True)
    for _sp_name, _sp in _a.spines.items():
        if _sp_name in ('top', 'right'):
            _sp.set_visible(False)   # open style; red 6h border removed (author 2026-09-04)
        # left/bottom: default black spine — suite bar-panel standard (matches scatters a-e + Figs 3/5/6)

    # (Aggregate-skill dashed reference line removed 2026-06-12: its 7.5% was the
    # best-model day-average from §4.1, but these bars are the all-model hourly
    # average, so the line confused more than it clarified. The bars + y-axis
    # already convey the overnight concentration; the 7.5% headline stays in §4.1.)
    if _i == 0:
        _a.set_ylabel('DL advantage (%) vs DLinear',
                      fontsize=FS_LABEL, labelpad=2)   # standard axis-label size (match scatters)
        # Night/Day zone labels removed per user (cleaner panel)

inline_panel_title(
    ax_f[0], 'b',
    'DL advantage over DLinear by target hour',   # plain label; the claim lives in §4.4 (2026-09-15); "target hour" as on the axis and in the caption (figure check 2026-10-02)
    pad_pts=14, gs_cell=outer[4, :],   # gaps 2026-10-04: (b) row is now row 4
)



# =====================================================================
# Save — production PNG + PDF via fig_config helper
# =====================================================================
savefig(fig, 'fig07_physics_mechanism', pad=0.05, bbox='fullwidth')