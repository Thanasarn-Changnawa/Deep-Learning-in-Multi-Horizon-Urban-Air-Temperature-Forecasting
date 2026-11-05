"""Standalone builder — extracted from generate_figures.ipynb cell 8.
Run directly: python fig_script/build_fig03.py

Figure 3: Multi-Horizon Forecast Comparison + Performance Degradation.
9-panel layout (Table 3 separated into a standalone table in §4.1):
  (a)-(e) Representative-day time-series per horizon (10-station mean ±1 std)
  (f)     MAE vs forecast horizon (network mean, 3-regime shading)
  (g)     Skill vs Persistence (%)
  (h)     Skill vs DLinear (%)

Merged from fig03_short_vs_long + fig04_performance_degradation (2026-04-21).
Revised 2026-04-21: removed Table 3 panel to avoid duplication with standalone
Table 3 in §4.1; that table now carries the *n DM-significance annotations.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pickle
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_config import (inline_panel_title, 
    FULL_W, DPI, MODEL_COLORS,
    REGIME_BG_COLORS, REGIME_COLOR, REGIME_NAMES,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE,
    apply_rcparams, savefig,
)
apply_rcparams()

_fd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'publication_results.json')
if not os.path.exists(_fd_path):
    _fd_path = 'publication_results.json'
try:
    with open(_fd_path, 'r', encoding='utf-8') as _f:
        fd = json.load(_f).get('figure_data', {})
except FileNotFoundError:
    fd = {}

_cache = pickle.load(open('ms_predictions_cache.pkl', 'rb'))
_ms_preds = _cache['ms_predictions']
_base_dir = '00_dataset/5min/20 stations'
ms_summary = fd.get('multistation', {}).get('summary', [])

PLOT_MODELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM', 'DLinear', 'Persistence']
DL_MODELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']
MODEL_LABELS_3 = {
    'PatchTST': 'PatchTST', 'iTransformer': 'iTransformer',
    'LSTM': 'LSTM', 'Attention-LSTM': 'Attn-LSTM',   # network-wide short form (Figs 4, 5, 6, 10, S3); review 2026-09-22
    'DLinear': 'DLinear', 'Persistence': 'Persistence',
}
LINE_STYLES = {
    'PatchTST': '-', 'iTransformer': '-', 'LSTM': '-',
    'Attention-LSTM': '-', 'DLinear': '--', 'Persistence': ':',
}
M_MARKERS = {'PatchTST': 'o', 'iTransformer': 's', 'LSTM': '^',
             'Attention-LSTM': 'D', 'DLinear': 'v', 'Persistence': 'X'}
M_LW = {'PatchTST': 1.8, 'iTransformer': 1.8, 'LSTM': 1.8,
        'Attention-LSTM': 1.8, 'DLinear': 1.4, 'Persistence': 1.2}
M_ALPHA = {'PatchTST': 0.9, 'iTransformer': 0.9, 'LSTM': 0.9,
           'Attention-LSTM': 0.9, 'DLinear': 0.85, 'Persistence': 0.55}

H_ORDER  = ['5min', '10min', '1h', '6h', '24h']
H_LABELS = {'5min': '5 min (t+1)', '10min': '10 min (t+2)', '1h': '1 h (t+12)',
            '6h': '6 h (t+72)', '24h': '24 h (t+288)'}
H_DISPLAY = ['5 min', '10 min', '1 h', '6 h', '24 h']
H_STEPS  = {'5min': 1, '10min': 2, '1h': 12, '6h': 72, '24h': 288}
panel_abc = list('abcde')
n_show = 288
# REP_DAY_IDX selects the day from the test-set common-start anchor.
# Common-start = 2017-05-25 02:05 (driven by station 10635588).
# REP_DAY_IDX = 1 -> 2017-05-26 (cleanest diurnal cycle inside the
# May 25-30 sub-window of Fig 9's May 15-30 climatology window;
# 16-18 °C amplitude across all stations, 5am trough / 15h peak).
REP_DAY_IDX = 1

# ============================================================================
# 9-panel figure: 2-column nested layout, a-e left, f-i right (panel i added 2026-09-26)
# ============================================================================
fig = plt.figure(figsize=(FULL_W, 7.30))   # +0.70 in for panel i under h (author 2026-09-24)  # canvas flattened in place (author-approved 2026-09-04) after reclaiming the
                                          # GridSpec margins below: panels keep
                                          # ~+10% height vs the old layout while
                                          # the tight-cropped total stays ~9.3 in
                                          # (same page footprint, no dead band).
outer = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.0], wspace=0.30,
                 top=0.99, bottom=0.075,  # reclaim default margins but keep room
                                          # for the x-axis labels above the legend
                 left=0.065, right=0.985) # fill FULL_W so the tight-cropped width
                                          # matches the other figures (same print
                                          # scale when Word fits all to 6.5 in)
left_gs  = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[0], hspace=0.38)
right_gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[1], hspace=0.38, height_ratios=[1, 1, 1, 0.95])   # f, g, h, i

ax_ts = [fig.add_subplot(left_gs[i, 0]) for i in range(5)]
ax_f  = fig.add_subplot(right_gs[0, 0])
ax_g  = fig.add_subplot(right_gs[1, 0])
ax_h  = fig.add_subplot(right_gs[2, 0])
ax_i  = fig.add_subplot(right_gs[3, 0])   # dual-benchmark margins with 95% intervals (author 2026-09-24/26)

# ============================================================================
# Panels (a)-(e): representative-day time-series
# ============================================================================
_stations = sorted(set(k[0] for k in _ms_preds.keys()))

_station_starts = {}
for sid in _stations:
    df_st = pd.read_csv(f'{_base_dir}/{sid}_5min_Interval.csv')
    ts = pd.to_datetime(df_st['Datetime'])
    temps = df_st['Temp'].values
    actuals_5 = _ms_preds[(sid, '5min', 'PatchTST')][1]
    target = np.array(actuals_5[:10])
    for i in range(len(temps) - 10):
        if np.allclose(temps[i:i+10], target, atol=0.01):
            _station_starts[sid] = ts.iloc[i]
            break

_common_start = max(_station_starts.values())
_day_offset_5min = REP_DAY_IDX * 288
_rep_day_anchor = _common_start + pd.Timedelta(days=REP_DAY_IDX)
REP_DAY_TAG = f'{_rep_day_anchor.day} {_rep_day_anchor:%b}'   # e.g. "26 May", the paper's date form (review 2026-09-22)
print(f'Fig 3 rep-day: {_rep_day_anchor}  ({REP_DAY_TAG})')

_global_ymin, _global_ymax = np.inf, -np.inf

for idx, h_label in enumerate(H_ORDER):
    ax = ax_ts[idx]
    h_shift = H_STEPS[h_label] - 1
    t_h = np.arange(n_show) * 5 / 60

    all_actuals = []
    all_preds = {mk: [] for mk in PLOT_MODELS}

    for sid in _stations:
        base_off = int((_common_start - _station_starts[sid]).total_seconds() / 300)
        h_off = base_off - h_shift + _day_offset_5min   # every station at the SAME target time; max(0, ...) pushed the last-opening station (base_off = 0) later by h_shift steps in panels b-e (found 2026-10-02)
        assert h_off >= 0, 'the representative day starts before the first target of station %s at %s' % (sid, h_label)
        h_len = len(_ms_preds[(sid, h_label, 'PatchTST')][1])
        if h_off + n_show > h_len:
            continue

        _, a_full = _ms_preds[(sid, h_label, 'PatchTST')]
        all_actuals.append(a_full[h_off:h_off + n_show])

        for mk in PLOT_MODELS:
            p_full, _ = _ms_preds[(sid, h_label, mk)]
            all_preds[mk].append(p_full[h_off:h_off + n_show])

    if not all_actuals:
        inline_panel_title(ax, panel_abc[idx], f'{H_LABELS[h_label]} ahead' + (f', {REP_DAY_TAG}' if idx == 0 else ''))   # date on panel a only (author 2026-09-15)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        continue

    act_stack = np.stack(all_actuals)
    act_mean = act_stack.mean(axis=0)
    act_std  = act_stack.std(axis=0, ddof=1)   # sample SD, as Table 2 / Section 3.4.2 (review 2026-09-22)

    ax.fill_between(t_h, act_mean - act_std, act_mean + act_std,
                    color='black', alpha=0.08, zorder=1)
    ax.plot(t_h, act_mean, color='black', linewidth=1.6, label='Observed', zorder=10)

    for mk in PLOT_MODELS:
        if not all_preds[mk]:
            continue
        p_stack = np.stack(all_preds[mk])
        p_mean = p_stack.mean(axis=0)
        p_std  = p_stack.std(axis=0, ddof=1)
        lw = 1.2 if mk not in ('Persistence', 'DLinear') else 0.9
        ls = LINE_STYLES.get(mk, '-')
        alpha = 0.55 if mk == 'Persistence' else 0.85
        ax.fill_between(t_h, p_mean - p_std, p_mean + p_std,
                        color=MODEL_COLORS[mk], alpha=0.07, zorder=2)
        ax.plot(t_h, p_mean, color=MODEL_COLORS[mk],
                linewidth=lw, linestyle=ls, alpha=alpha,
                label=MODEL_LABELS_3[mk], zorder=5)

    _ymin, _ymax = ax.get_ylim()
    _global_ymin = min(_global_ymin, _ymin)
    _global_ymax = max(_global_ymax, _ymax)

    inline_panel_title(ax, panel_abc[idx], f'{H_LABELS[h_label]} ahead' + (f', {REP_DAY_TAG}' if idx == 0 else ''))   # project-standard title; date on panel a only (author 2026-09-15)
    ax.set_xlim(0, 24)
    _x0 = _rep_day_anchor.hour + _rep_day_anchor.minute / 60     # the window opens at 02:05 local time, so x = hours after 02:05
    ax.set_xticks([c - _x0 for c in (6, 12, 18, 24)])              # ticks at clock hours, as in Figs. 7b and 9
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)     # open style (user decision 2026-06-11)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.4)
    if idx == 4:
        ax.set_xticklabels(['06', '12', '18', '24'])
        _end = _rep_day_anchor + pd.Timedelta(minutes=5 * (n_show - 1))   # the window runs past midnight (02:05 to 02:00 next day)
        ax.set_xlabel(f'Local time, {_rep_day_anchor.day}–{_end.day} {_rep_day_anchor:%b} (h)' if _end.day != _rep_day_anchor.day
                      else f'Local time on {REP_DAY_TAG} (h)', fontsize=FS_TICK)   # "26–27 May" (author 2026-10-02)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('T (\u00b0C)', fontsize=FS_TICK)

for _ax in ax_ts:
    _ax.set_ylim(_global_ymin, _global_ymax)


# ============================================================================
# Panels (f)-(h): performance-degradation curves
# ============================================================================
def _val(model, horizon, metric='MAE'):
    ms_df = pd.DataFrame(ms_summary) if isinstance(ms_summary, list) else ms_summary
    r = ms_df[(ms_df['Horizon'] == horizon) & (ms_df['Model'] == model)]
    mc = f'{metric}_mean'
    if len(r) > 0 and mc in r.columns:
        return float(r[mc].values[0])
    return np.nan


ms_df_5 = pd.DataFrame(ms_summary) if isinstance(ms_summary, list) else ms_summary
x_pos = np.arange(len(H_ORDER))

_ps_sd = pd.read_csv('multistation_perstation_metrics.csv')   # per-station MAEs: SD recomputed with ddof=1 (the summary's MAE_std is the population SD; review 2026-09-22)
mae_data = {m: [] for m in PLOT_MODELS}
mae_std_d = {m: [] for m in PLOT_MODELS}
skill_pers = {m: [] for m in PLOT_MODELS}
skill_dlin = {m: [] for m in DL_MODELS}

for h in H_ORDER:
    pers_mae = _val('Persistence', h)
    dlin_mae = _val('DLinear', h)
    for m in PLOT_MODELS:
        mae_m = _val(m, h)
        r = ms_df_5[(ms_df_5['Horizon'] == h) & (ms_df_5['Model'] == m)]
        _v = _ps_sd[(_ps_sd['Horizon'] == h) & (_ps_sd['Model'] == m)]['MAE'].values
        mae_s = float(np.std(_v, ddof=1)) if len(_v) > 1 else (float(r['MAE_std'].values[0]) if len(r) > 0 else 0)
        mae_data[m].append(mae_m)
        mae_std_d[m].append(mae_s)
        if not np.isnan(mae_m) and pers_mae > 0:
            skill_pers[m].append((1 - mae_m / pers_mae) * 100)
        else:
            skill_pers[m].append(np.nan)
    for m in DL_MODELS:
        mae_m = _val(m, h)
        if not np.isnan(mae_m) and dlin_mae > 0:
            skill_dlin[m].append((1 - mae_m / dlin_mae) * 100)
        else:
            skill_dlin[m].append(np.nan)

for ax in [ax_f, ax_g, ax_h, ax_i]:
    # regime background fills removed (2026-06-11, user decision: cleaner look);
    # the dashed boundaries + colored labels carry the regime separation
    ax.axvline(2.5, color='#aaaaaa', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.axvline(3.5, color='#aaaaaa', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)     # open style (user decision 2026-06-11)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(x_pos)

# Marker draw order of panels f and g (figure check 2026-10-02): the iTransformer square lowest and the PatchTST circle next,
# so that the circle is no longer buried under the square where the two coincide (24 h); every other marker keeps its place on top.
M_Z = {'iTransformer': 5.8, 'PatchTST': 5.9}

# ---- Panel (f): MAE ----
for m in PLOT_MODELS:
    means = np.array(mae_data[m])
    stds = np.array(mae_std_d[m])
    ax_f.fill_between(x_pos, means - stds, means + stds,
                      color=MODEL_COLORS[m], alpha=0.08)
    ax_f.plot(x_pos, means, color=MODEL_COLORS[m], linewidth=M_LW[m],
              linestyle=LINE_STYLES[m], alpha=M_ALPHA[m], zorder=5)
    ax_f.scatter(x_pos, means, color=MODEL_COLORS[m], marker=M_MARKERS[m],
                 s=26, zorder=M_Z.get(m, 6), edgecolors='white', linewidths=0.4)

ax_f.set_ylim(0, 4.2)   # tight top: regime labels live at the bottom of the bands

ax_f.text(1.0, -0.04, f'Regime 1\n{REGIME_NAMES[1]}',
          transform=ax_f.get_xaxis_transform(),
          ha='center', va='top', fontsize=FS_DENSE, fontstyle='italic',
          color=REGIME_COLOR[1])
def _stack_regime(name):
    # narrow bands (regimes 2-3): break the name at its last space so the
    # text block stays inside its shaded span (fig_config stays the source)
    return '\n'.join(name.rsplit(' ', 1))

ax_f.text(3.0, -0.04, f'Regime 2\n{_stack_regime(REGIME_NAMES[2])}',
          transform=ax_f.get_xaxis_transform(),
          ha='center', va='top', fontsize=FS_DENSE, fontstyle='italic',
          color=REGIME_COLOR[2])
ax_f.text(4.0, -0.04, f'Regime 3\n{_stack_regime(REGIME_NAMES[3])}',
          transform=ax_f.get_xaxis_transform(),
          ha='center', va='top', fontsize=FS_DENSE, fontstyle='italic',
          color=REGIME_COLOR[3])

pers_6h = mae_data['Persistence'][3]
ax_f.annotate('R\u00b2 = \u22120.50', xy=(3, pers_6h),
              xytext=(2.70, pers_6h - 0.45), ha='right',
              fontsize=FS_DENSE, color=MODEL_COLORS['Persistence'], fontweight='bold',
              arrowprops=dict(arrowstyle='->', color=MODEL_COLORS['Persistence'],
                              linewidth=0.6, shrinkB=3), zorder=15)

ax_f.set_ylabel('MAE (\u00b0C)', fontsize=FS_TICK)
inline_panel_title(ax_f, 'f', 'MAE vs forecast horizon')
ax_f.set_xticklabels([])

# ---- Panel (g): Skill vs Persistence ----
ax_g.axhline(0, color='black', linewidth=0.5, alpha=0.4, zorder=1)
for m in PLOT_MODELS:
    if m == 'Persistence':
        ax_g.plot(x_pos, [0]*5, color=MODEL_COLORS[m], linewidth=M_LW[m],
                  linestyle=LINE_STYLES[m], alpha=M_ALPHA[m], zorder=3)
        ax_g.scatter(x_pos, [0]*5, color=MODEL_COLORS[m], marker=M_MARKERS[m],
                     s=22, zorder=4, edgecolors='white', linewidths=0.4)
        continue
    means = np.array(skill_pers[m])
    ax_g.plot(x_pos, means, color=MODEL_COLORS[m], linewidth=M_LW[m],
              linestyle=LINE_STYLES[m], alpha=M_ALPHA[m], zorder=5)
    ax_g.scatter(x_pos, means, color=MODEL_COLORS[m], marker=M_MARKERS[m],
                 s=26, zorder=M_Z.get(m, 6), edgecolors='white', linewidths=0.4)
ax_g.set_ylabel('Skill vs Persistence (%)', fontsize=FS_TICK)
inline_panel_title(ax_g, 'g', 'Skill vs Persistence')
ax_g.set_xticklabels([])

# ---- Panel (h): Skill vs DLinear ----
ax_h.axhline(0, color=MODEL_COLORS['DLinear'], linewidth=0.9, alpha=0.6,
             linestyle='--', zorder=1)
for m in DL_MODELS:
    means = np.array(skill_dlin[m])
    ax_h.plot(x_pos, means, color=MODEL_COLORS[m], linewidth=M_LW[m],
              linestyle=LINE_STYLES[m], alpha=M_ALPHA[m], zorder=5)
    ax_h.scatter(x_pos, means, color=MODEL_COLORS[m], marker=M_MARKERS[m],
                 s=28, zorder=6, edgecolors='white', linewidths=0.4)

# whisker per horizon = exact station-level 95% bootstrap interval of the best candidate's gain over DLinear
# (fig03_intervals.csv from compute_fig03_intervals.py; same construction as Table S8; author 2026-09-24)
_ci = pd.read_csv('fig03_intervals.csv'); _cih = _ci[_ci.quantity == 'gain_best'].set_index('horizon')
for _i, _hz in enumerate(H_ORDER):
    _lo, _hi = float(_cih.loc[_hz, 'A_lo']), float(_cih.loc[_hz, 'A_hi'])
    ax_h.plot([_i, _i], [_lo, _hi], color='#333333', linewidth=0.8, zorder=5.5)
    for _y in (_lo, _hi): ax_h.plot([_i - 0.08, _i + 0.08], [_y, _y], color='#333333', linewidth=0.8, zorder=5.5)

_valid_dl = [skill_dlin[m][3] for m in DL_MODELS if not np.isnan(skill_dlin[m][3])]
if _valid_dl:
    min_dl_6h = min(_valid_dl); max_dl_6h = max(_valid_dl)
    ax_h.text(3.0, max_dl_6h + 4.2, f'All four candidates > 0\n(+{min_dl_6h:.1f}% to +{max_dl_6h:.1f}%)',
              fontsize=FS_DENSE, fontweight='bold', color='#333333', ha='center', va='bottom', zorder=15)   # no arrow: the note is about all four markers (QA 2026-09-26)

ax_h.set_ylabel('Skill vs DLinear (%)', fontsize=FS_TICK)
# key for the whisker (same wording as panel i), top-left in axes coordinates (QA 2026-09-26)
ax_h.plot([0.025, 0.025], [0.905, 0.985], transform=ax_h.transAxes, color='#333333', linewidth=0.8, clip_on=False)
ax_h.text(0.045, 0.945, 'whisker: best candidate, 95% CI', transform=ax_h.transAxes, fontsize=FS_DENSE, va='center', color='#333333')
inline_panel_title(ax_h, 'h', 'Skill vs DLinear')
ax_h.set_xticks(x_pos)
ax_h.set_xticklabels([])   # tick labels and the x label now sit under panel i

# ---- Panel (i): the rule made visible (author 2026-09-26, Form C) ----
# Each dot = one dual-benchmark win (candidate at a station, lower MAE than both benchmarks AND Holm-corrected DM
# significant against both), placed at its margin in degC over the benchmark it beats by less (the quantity the
# Section 4.1 floor sweep tests); position against the dashed 0.21 degC line says whether it matters. Whisker = network-mean margin of the
# best candidate with its exact station-level 95% interval (fig03_intervals.csv, quantity margin_best).
SENSOR_ACC = 0.21
_dm_i = pd.read_csv('multistation_dm_tests_corrected.csv'); _pm_i = pd.read_csv('multistation_perstation_metrics.csv')
_mae_i = _pm_i.set_index(['Station', 'Horizon', 'Model'])['MAE']
def _sig_i(st, hz, model, bench):
    r = _dm_i[(_dm_i.Station == st) & (_dm_i.Horizon == hz) & (_dm_i.Comparison == f'{model} vs {bench}')]
    return len(r) > 0 and r.Significant_corrected.iloc[0] == 'Yes' and r.DM_Stat.iloc[0] < 0
_wins_i = {hz: [] for hz in H_ORDER}
for _hz in H_ORDER:
    for _st in sorted(_pm_i.Station.unique()):
        for _c in DL_MODELS:
            _mm, _md, _mp = _mae_i[(_st, _hz, _c)], _mae_i[(_st, _hz, 'DLinear')], _mae_i[(_st, _hz, 'Persistence')]
            if _mm < _md and _mm < _mp and _sig_i(_st, _hz, _c, 'DLinear') and _sig_i(_st, _hz, _c, 'Persistence'):
                _wins_i[_hz].append((_c, min(_md - _mm, _mp - _mm)))
print('Fig 3i dual-benchmark wins per horizon:', [len(_wins_i[h]) for h in H_ORDER], '(Table 2: 1/16/10/12/2)')
ax_i.axhline(SENSOR_ACC, color='#333333', linewidth=0.8, linestyle='--', zorder=2)
ax_i.axhline(0, color='black', linewidth=0.5, zorder=1)
_rng_i = np.random.default_rng(5)
for _i, _hz in enumerate(H_ORDER):
    _v = np.array([w[1] for w in _wins_i[_hz]]); _mods = [w[0] for w in _wins_i[_hz]]; _xs = _i + _rng_i.uniform(-0.22, 0.22, len(_v))   # wins scattered around the centred whisker
    for _c in DL_MODELS:   # dots carry the model: marker and colour as in the figure legend
        _sel = np.array([mm == _c for mm in _mods], dtype=bool)
        if _sel.any(): ax_i.scatter(_xs[_sel], _v[_sel], s=11, c=MODEL_COLORS[_c], marker=M_MARKERS[_c], alpha=0.65, edgecolors='none', zorder=3)
    _r = _ci[(_ci.horizon == _hz) & (_ci.quantity == 'margin_best')].iloc[0]; _x = _i
    ax_i.plot([_x, _x], [_r['A_lo'], _r['A_hi']], color='black', linewidth=1.5, zorder=5)
    for _y in (_r['A_lo'], _r['A_hi']): ax_i.plot([_x - 0.07, _x + 0.07], [_y, _y], color='black', linewidth=1.0, zorder=5)
    ax_i.plot(_x, _r['point'], 'o', color='white', markeredgecolor='black', markeredgewidth=1.0, markersize=3.8, zorder=6)
    _lab = 'n = %d win%s' % (len(_v), '' if len(_v) == 1 else 's')
    _lab += chr(10) + '○ ' + ('%.2f' % _r['point']).replace('-', '−')   # network-mean margin under the count (author 2026-09-26); the open-circle glyph ties the number to the circle of the key, so it is not read as the mean of the counted wins (figure check 2026-10-02)
    ax_i.text(_i, 0.435, _lab, fontsize=FS_DENSE, ha='center', va='top', color='#333333', linespacing=1.05)
ax_i.text(-0.42, SENSOR_ACC + 0.006, 'sensor accuracy 0.21 \u00b0C', fontsize=FS_DENSE, ha='left', va='bottom', color='#333333')
# short two-line key (author 2026-09-26); the caption carries the full definitions
ax_i.plot([-0.38, -0.38], [0.320, 0.350], color='black', linewidth=1.5, clip_on=False); ax_i.plot(-0.38, 0.335, 'o', color='white', markeredgecolor='black', markeredgewidth=1.0, markersize=3.8, clip_on=False)
ax_i.text(-0.25, 0.335, 'best candidate, 10-station mean, 95% CI', fontsize=FS_DENSE, va='center', color='#333333')
ax_i.set_ylim(-0.04, 0.45); ax_i.set_yticks([0.0, 0.1, 0.2, 0.3])   # no gridline through the count labels (QA 2026-09-26); ax_i.set_xlim(-0.5, 4.5)
ax_i.set_xticks(x_pos); ax_i.set_xticklabels(H_DISPLAY, fontsize=FS_TICK); ax_i.set_xlabel('Forecast horizon', fontsize=FS_TICK)
ax_i.set_ylabel('Margin over benchmark (\u00b0C)', fontsize=FS_TICK); ax_i.tick_params(axis='y', labelsize=FS_TICK)
inline_panel_title(ax_i, 'i', 'Dual-benchmark margin, 95% CI')


# ============================================================================
# Inline per-panel skill annotations on time-series panels (a)-(e)
# Best DL candidate, skill-% vs both benchmarks, dual-benchmark station count
# ============================================================================
_dm_df_anno = pd.read_csv('multistation_dm_tests_corrected.csv')
_ps_df_anno = pd.read_csv('multistation_perstation_metrics.csv')

_sig_stations_per_h = {}
for _h in H_ORDER:
    _stations_sig = set()
    _dm_h = _dm_df_anno[_dm_df_anno['Horizon'] == _h]
    _ps_h = _ps_df_anno[_ps_df_anno['Horizon'] == _h]
    for _sid in _ps_h['Station'].unique():
        for _m in DL_MODELS:
            _c1 = _dm_h[(_dm_h['Station'] == _sid) & (_dm_h['Comparison'] == f'{_m} vs DLinear')]
            _sig_dl = (len(_c1) > 0 and _c1['Significant_corrected'].values[0] == 'Yes'
                       and float(_c1['DM_Stat'].values[0]) < 0)
            _c2 = _dm_h[(_dm_h['Station'] == _sid) & (_dm_h['Comparison'] == f'{_m} vs Persistence')]
            _sig_per = (len(_c2) > 0 and _c2['Significant_corrected'].values[0] == 'Yes'
                        and float(_c2['DM_Stat'].values[0]) < 0)
            if not (_sig_dl and _sig_per):
                continue
            _mr = _ps_h[(_ps_h['Station'] == _sid) & (_ps_h['Model'] == _m)]
            _dl_r = _ps_h[(_ps_h['Station'] == _sid) & (_ps_h['Model'] == 'DLinear')]
            _per_r = _ps_h[(_ps_h['Station'] == _sid) & (_ps_h['Model'] == 'Persistence')]
            if len(_mr) > 0 and len(_dl_r) > 0 and len(_per_r) > 0:
                if (float(_mr['MAE'].values[0]) < float(_dl_r['MAE'].values[0]) and
                    float(_mr['MAE'].values[0]) < float(_per_r['MAE'].values[0])):
                    _stations_sig.add(_sid)
                    break
    _sig_stations_per_h[_h] = len(_stations_sig)

for _idx, _h in enumerate(H_ORDER):
    ax = ax_ts[_idx]
    _best_dl = min(DL_MODELS, key=lambda m: mae_data[m][_idx])
    s_p = skill_pers[_best_dl][_idx]
    s_d = skill_dlin[_best_dl][_idx]
    best_label = MODEL_LABELS_3.get(_best_dl, _best_dl)
    best_mae = mae_data[_best_dl][_idx]
    best_std = mae_std_d[_best_dl][_idx]
    _mfmt = lambda v: ('%+.1f' % v).replace('-', '\u2212')   # true minus sign in the box skills (figure check 2026-10-07)
    ann = (f'Best candidate: {best_label} ({best_mae:.2f}\u00b1{best_std:.2f} \u00b0C)\n'   # "candidate", the paper's word (was "Best DL"; figure check 2026-10-02)
           f'{_mfmt(s_p)}% vs Persistence, {_mfmt(s_d)}% vs DLinear')   # 'Persistence' spelled out as in the legend (2026-10-07)
    ax.text(0.98, 0.05, ann, transform=ax.transAxes, fontsize=FS_DENSE,
            va='bottom', ha='right', fontstyle='italic', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='0.7', linewidth=0.4))


# ============================================================================
# Shared legend at bottom
# ============================================================================
legend_elements = [
    Line2D([0], [0], color='black', linewidth=1.6, label='Observed'),
]
for m in PLOT_MODELS:
    legend_elements.append(
        Line2D([0], [0], marker=M_MARKERS[m], color=MODEL_COLORS[m],
               linewidth=M_LW[m], linestyle=LINE_STYLES[m],
               markerfacecolor=MODEL_COLORS[m], markersize=5,
               markeredgecolor='white', markeredgewidth=0.4,
               label=MODEL_LABELS_3[m], alpha=M_ALPHA[m]))

fig.legend(handles=legend_elements, loc='lower center', ncol=7,
           fontsize=FS_DENSE, frameon=False,
           bbox_to_anchor=(0.5, -0.0038),  # gaps 2026-10-04: legend lifted 0.177 in (0.0242 x 7.30) so the legend gap is 0.08 in
           handlelength=1.8, columnspacing=1.0, handletextpad=0.4)

savefig(fig, 'fig03', bbox='fullwidth')  # exact FULL_W width for consistent print scale
print('Saved fig03.png + fig03.pdf (9-panel layout: a-e, f-i; panel i added 2026-09-26)')