# -*- coding: utf-8 -*-
"""Supplement Fig. S2: observed vs predicted air temperature, six models x five horizons, one station on one day.

Run from the project root:  python fig_script/build_figS02.py   ->  figS12_multi_horizon_predictions.png (+ .pdf)

Ported on 2 Oct 2026 from generate_figures.ipynb, cell 22 (that cell is now superseded; do not run it), with three fixes:
  * every column is cut at the SAME target times. The cell took each horizon's array from its own midpoint (n // 2), so the
    five columns showed different windows (5 June 03:00, 03:05, 03:30, 06:00 and 15:00). Index j of the H-step series is the
    target t0 + (H - 1 + j) steps, t0 = 13 May 2017 15:45 (first 5-min target), so the 5-min index i maps to j = i - (H - 1);
  * the day shown is 26 May 2017, 00:00-23:55, the day of Fig. 3a-e (the author chose this on 2 Oct 2026), and the axis
    names it (the cell printed "Hour of day (21 June)");
  * one x-axis label for the grid and the legend below it (it covered the title of the 24 h column).
12 Oct 2026: the observed line and its legend swatch are drawn in black, as in Figs. 3a-e, 8d and 9c (they were PatchTST blue,
so in the PatchTST row observed and predicted differed only by dash pattern).
Data: publication_results.json, figure_data.all_model_predictions = the reference station, 10th St Bridge (10635576).
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fig_config import *
apply_rcparams()
palette = PALETTE
with open('publication_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
fd = data['figure_data']
# === Figure S12: Full Multi-Horizon Prediction Grid (All Models × 5 Horizons) ===
# This is the detailed version of Figure 3. Main text uses the 2-panel summary.
amp = fd.get('all_model_predictions', {})
mh_preds = fd.get('mh_predictions', {})

PLOT_MODELS  = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM', 'DLinear', 'Persistence']
MODEL_LABELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM', 'DLinear', 'Persistence']
horizons_plot   = ['5min', '10min', '1h', '6h', '24h']
horizon_titles  = ['5 min ahead', '10 min ahead', '1 h ahead', '6 h ahead', '24 h ahead']   # 17 Oct 2026: the paper's horizon form (Figs. 3, 4, 8); was '5-min ahead' ... '24-hour ahead'

window_size = 288  # 24-hour display window at 5-min resolution
H_STEPS_S2 = {'5min': 1, '10min': 2, '1h': 12, '6h': 72, '24h': 288}
n_rows = len(PLOT_MODELS)
n_cols = len(horizons_plot)
ROW_H  = 1.65
SUPY_X = float(os.environ.get('FIGS02_SUPY_X', '0.012'))   # figure-x of the 'T (°C)' label, left of the longest row name

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(FULL_W, ROW_H * n_rows),
                         sharex=True)

# ── First pass: collect GLOBAL y-range across all panels ──
global_ymin, global_ymax = np.inf, -np.inf
for row, model_key in enumerate(PLOT_MODELS):
    ymin, ymax = np.inf, -np.inf
    for col, h_label in enumerate(horizons_plot):
        actuals_full = None
        if h_label in amp and 'actuals' in amp[h_label]:
            actuals_full = np.array(amp[h_label]['actuals'])
        elif h_label in mh_preds and 'actuals' in mh_preds[h_label]:
            actuals_full = np.array(mh_preds[h_label]['actuals'])
        if actuals_full is None:
            continue
        preds_full = None
        if h_label in amp and model_key in amp[h_label]:
            preds_full = np.array(amp[h_label][model_key])
        elif h_label in mh_preds:
            pk = model_key + '_preds'
            if pk in mh_preds[h_label]:
                preds_full = np.array(mh_preds[h_label][pk])
            elif model_key == 'PatchTST' and 'preds' in mh_preds[h_label]:
                preds_full = np.array(mh_preds[h_label]['preds'])
        if preds_full is None:
            continue
        n = min(len(actuals_full), len(preds_full))
        start = 3555 - (H_STEPS_S2[h_label] - 1)   # same target times in every column (fixed 2026-10-02; was n // 2 per column)
        assert start >= 0 and start + window_size <= n
        end = start + window_size
        vals = np.concatenate([actuals_full[start:end], preds_full[start:end]])
        ymin = min(ymin, vals.min())
        ymax = max(ymax, vals.max())
    if ymin < np.inf:
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)

g_margin = (global_ymax - global_ymin) * 0.05
global_ylim = (global_ymin - g_margin, global_ymax + g_margin)

# ── Second pass: plot ──
for row, (model_key, model_label) in enumerate(
        zip(PLOT_MODELS, MODEL_LABELS)):
    for col, h_label in enumerate(horizons_plot):
        ax = axes[row, col]

        actuals_full = None
        if h_label in amp and 'actuals' in amp[h_label]:
            actuals_full = np.array(amp[h_label]['actuals'])
        elif h_label in mh_preds and 'actuals' in mh_preds[h_label]:
            actuals_full = np.array(mh_preds[h_label]['actuals'])

        if actuals_full is None:
            ax.set_visible(False)
            continue

        preds_full = None
        if h_label in amp and model_key in amp[h_label]:
            preds_full = np.array(amp[h_label][model_key])
        elif h_label in mh_preds:
            pk = model_key + '_preds'
            if pk in mh_preds[h_label]:
                preds_full = np.array(mh_preds[h_label][pk])
            elif model_key == 'PatchTST' and 'preds' in mh_preds[h_label]:
                preds_full = np.array(mh_preds[h_label]['preds'])

        if preds_full is None:
            ax.set_facecolor('#f5f5f5')
            ax.text(0.5, 0.5, 'Not evaluated', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, fontstyle='italic',
                    color='#999999')
            ax.set_yticks([])
            ax.tick_params(left=False)
            if row == 0:
                ax.set_title(horizon_titles[col], fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(model_label, fontsize=8, fontweight='bold',
                              rotation=0, ha='right', va='center', labelpad=8)
            if row == n_rows - 1:
                pass
            continue

        n = min(len(actuals_full), len(preds_full))
        start = 3555 - (H_STEPS_S2[h_label] - 1)   # same target times in every column (fixed 2026-10-02; was n // 2 per column)
        assert start >= 0 and start + window_size <= n
        end = start + window_size
        actuals = actuals_full[start:end]
        preds = preds_full[start:end]
        t = np.arange(len(actuals)) * 5 / 60  # 5-min steps → hours

        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds)**2))

        ax.plot(t, actuals, color='black', lw=0.9, label='Observed')   # black, as in Figs. 3a-e, 8d and 9c (12 Oct 2026; was PatchTST blue)
        m_color = MODEL_COLORS.get(model_key, palette[1])
        ax.plot(t, preds, color=m_color, lw=0.8, ls=(0, (4, 2)),
                label='Predicted')
        ax.fill_between(t, actuals, preds, alpha=0.18, color=m_color)

        ax.set_xlim(0, 24)
        ax.set_ylim(global_ylim)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15, linewidth=0.3)

        # Metric badge: upper-left since 17 Oct 2026 (at the upper right it covered the curves from about 13.5 h to 18.4 h,
        # the 26 May maximum, in every panel; the upper-left corner, 0-8 h above 23 °C, is empty in all 30 panels)
        ax.text(0.03, 0.95,
                f'MAE={mae:.2f}\nRMSE={rmse:.2f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=5.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.85))

        if row == 0:
            ax.set_title(horizon_titles[col], fontsize=9, fontweight='bold')
        if col == 0:
            ax.set_ylabel(model_label, fontsize=8, fontweight='bold',
                          rotation=0, ha='right', va='center', labelpad=8)
        if row == n_rows - 1:
            pass

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# Legend
handles = [plt.Line2D([0], [0], color='black', lw=1.0, label='Observed'),
           plt.Line2D([0], [0], color='#666666', lw=0.8, ls=(0, (4, 2)), label='Predicted (model color)')]
plt.tight_layout(h_pad=0.3, w_pad=0.5)
fig.subplots_adjust(top=0.97, left=0.10, bottom=0.07)
fig.supxlabel('Local time on 26 May (h)', fontsize=8, y=0.035)
fig.supylabel('T (°C)', fontsize=8, x=SUPY_X)   # 17 Oct 2026: the temperature unit, as on Fig. 3a-e (the per-axis slot holds the model name)
fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=7,
           frameon=True, edgecolor='0.85', fancybox=False,
           bbox_to_anchor=(0.55, 0.025))   # below the grid: it covered the title of the 24 h column (2026-10-02)
plt.savefig('figS12_multi_horizon_predictions.png', dpi=DPI, bbox_inches='tight')
plt.savefig('figS12_multi_horizon_predictions.pdf', dpi=DPI, bbox_inches='tight')
plt.close()

print(f'Fig. S2 saved: {n_rows} models x {n_cols} horizons')
