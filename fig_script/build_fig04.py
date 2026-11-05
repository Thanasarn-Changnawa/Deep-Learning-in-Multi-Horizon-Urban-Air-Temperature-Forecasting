"""Fig 5: LCZ-rooted hierarchy — extracted from generate_figures.ipynb cell 10."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import sys, io, pickle as _pkl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib.colors as mcolors
from fig_config import (
    FULL_W, DPI, LC_COLORS, LCZ_COLORS, MODEL_COLORS, MODEL_SHORT,
    CANDIDATES, ORDERED_STATIONS_LCZ, SHORT_NAMES, LCZ_SHORT,
    VEG_COLORS, VEG_ORDER, HORIZONS, HZ_LABELS, DENDRO,
    FS_LABEL, FS_TICK, FS_LEGEND, FS_DENSE, FS_DENSE_SM,
    LW_ROOT, LW_BRANCH, LW_LEAF, PAD_VEG, PAD_LCZ, PAD_LC,
    COL_ROOT, COL_CONNECTOR, BLW_VEG, BLW_LCZ, BLW_LC,
    HSPACE_PANELS, WSPACE_PANELS,
    MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT,
    apply_rcparams,
)
apply_rcparams()

# Candidate subset of MODEL_COLORS
CAND_COLORS = {m: MODEL_COLORS[m] for m in CANDIDATES}

ps_df = pd.read_csv('multistation_perstation_metrics.csv')
dm_df = pd.read_csv('multistation_dm_tests_corrected.csv')

n_sta = len(ORDERED_STATIONS_LCZ)
n_mod = len(CANDIDATES)
station_labels = [SHORT_NAMES.get(sn, sn) for _, sn, _, _, _ in ORDERED_STATIONS_LCZ]
lc_list = [lc for _, _, lc, _, _ in ORDERED_STATIONS_LCZ]
lcz_list = [lcz for _, _, _, lcz, _ in ORDERED_STATIONS_LCZ]
veg_list = [veg for _, _, _, _, veg in ORDERED_STATIONS_LCZ]

# Y-positions with gaps: Surface type (largest) > LCZ > LC
GAP_VEG = 0.0
GAP_LCZ = 0.0
GAP_LC = 0.0
y_pos = []
current_y = 0
prev_veg = None; prev_lcz = None; prev_lc = None
for i in range(n_sta):
    veg = veg_list[i]; lcz = lcz_list[i]; lc = lc_list[i]
    if prev_veg is not None:
        if veg != prev_veg:
            current_y += GAP_VEG
        elif lcz != prev_lcz:
            current_y += GAP_LCZ
        elif lc != prev_lc:
            current_y += GAP_LC
    y_pos.append(current_y)
    current_y += 1
    prev_veg = veg; prev_lcz = lcz; prev_lc = lc
y_pos = np.array(y_pos)
y_max = y_pos[-1]

# Build 4-level tree: Surface type -> LCZ -> LC -> Station
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

def check_sig_both(dm_h, sid, model):
    c1 = dm_h[(dm_h['Station'] == sid) & (dm_h['Comparison'] == f'{model} vs DLinear')]
    sig_dl = (len(c1) > 0 and c1['Significant_corrected'].values[0] == 'Yes'
              and float(c1['DM_Stat'].values[0]) < 0)
    c2 = dm_h[(dm_h['Station'] == sid) & (dm_h['Comparison'] == f'{model} vs Persistence')]
    sig_per = (len(c2) > 0 and c2['Significant_corrected'].values[0] == 'Yes'
               and float(c2['DM_Stat'].values[0]) < 0)
    return sig_dl and sig_per

# Build matrices
sig_matrix = {}; skill_matrix = {}; beats_mae_matrix = {}
sig_summary = {hz: {m: 0 for m in CANDIDATES} for hz in HORIZONS}
for hz in HORIZONS:
    dh = ps_df[ps_df['Horizon'] == hz]
    dm_h = dm_df[dm_df['Horizon'] == hz]
    sig_arr = np.zeros((n_sta, n_mod), dtype=bool)
    beats_arr = np.zeros((n_sta, n_mod), dtype=bool)
    skill_arr = np.full((n_sta, n_mod), np.nan)
    for i, (sid, sn, lc, lcz, _veg) in enumerate(ORDERED_STATIONS_LCZ):
        dl_r = dh[(dh['Station'] == sid) & (dh['Model'] == 'DLinear')]
        per_r = dh[(dh['Station'] == sid) & (dh['Model'] == 'Persistence')]
        mae_dl = float(dl_r['MAE'].values[0]) if len(dl_r) > 0 else np.nan
        mae_per = float(per_r['MAE'].values[0]) if len(per_r) > 0 else np.nan
        for j, m in enumerate(CANDIDATES):
            mr = dh[(dh['Station'] == sid) & (dh['Model'] == m)]
            if len(mr) > 0:
                mae_m = float(mr['MAE'].values[0])
                skill_arr[i, j] = (1 - mae_m / mae_dl) * 100 if mae_dl > 0 else np.nan
                skill_per = (1 - mae_m / mae_per) * 100 if mae_per > 0 else np.nan
                beats_arr[i, j] = mae_m < mae_dl
                sig_arr[i, j] = check_sig_both(dm_h, sid, m) and skill_arr[i, j] > 0 and skill_per > 0
                if sig_arr[i, j]:
                    sig_summary[hz][m] += 1
    sig_matrix[hz] = sig_arr
    skill_matrix[hz] = skill_arr
    beats_mae_matrix[hz] = beats_arr

# Build ns_summary: stations that beat DLinear in MAE but NOT sig
ns_summary = {hz: {m: 0 for m in CANDIDATES} for hz in HORIZONS}
total_wins_summary = {hz: {m: 0 for m in CANDIDATES} for hz in HORIZONS}
for hz in HORIZONS:
    for i in range(n_sta):
        for j, m in enumerate(CANDIDATES):
            if beats_mae_matrix[hz][i, j]:
                total_wins_summary[hz][m] += 1
                if not sig_matrix[hz][i, j]:
                    ns_summary[hz][m] += 1

# ---- FIGURE ----
fig = plt.figure(figsize=(FULL_W, 7.5))
gs = gridspec.GridSpec(3, 7,
                       height_ratios=[5.0, 1.0, 1.0],
                       width_ratios=[4.0, 1, 1, 1, 1, 1, 0.8],
                       hspace=0.15, wspace=WSPACE_PANELS)  # tighter hspace for 3-row layout
fig.subplots_adjust(left=0.07, right=MARGIN_RIGHT, top=0.88, bottom=0.04)  # left=0.07 for dendrogram, top=0.88 for regime labels

ax_tile0 = fig.add_subplot(gs[0, 1])
ax_tile0.set_ylim(y_max + 0.5, -0.5)

ax_tree = fig.add_subplot(gs[0, 0], sharey=ax_tile0)
ax_tree.set_xlim(0, DENDRO['XLIM'])

# ---- 4-LEVEL DENDROGRAM: Surface type -> LCZ -> LC -> Station ----
# Unpack DENDRO positions
VEG_X2 = DENDRO['VEG_X2']
J0_X   = DENDRO['J0_X']
LCZ_X2 = DENDRO['LCZ_X2']
J1_X   = DENDRO['J1_X']
LC_X2  = DENDRO['LC_X2']
J2_X   = DENDRO['J2_X']
STA_X  = DENDRO['STA_X']
VEG_R  = DENDRO['VEG_R']
LCZ_R  = DENDRO['LCZ_R']
LC_R   = DENDRO['LC_R']
VEG_BW = 2.0
LCZ_BW = 1.5
LC_BW  = 1.95

for veg, lcz_groups in full_tree:
    veg_col = VEG_COLORS[veg]
    all_veg_idx = []
    for lcz_val, lc_groups in lcz_groups:
        for lc_val, sta_idx in lc_groups:
            all_veg_idx.extend(sta_idx)
    veg_ys = y_pos[all_veg_idx]
    veg_mid = (veg_ys.min() + veg_ys.max()) / 2

    # Surface type label
    veg_short = 'Non-veg.' if veg == 'Non-vegetated' else 'Vegetated'
    ax_tree.text(VEG_R - 0.15, veg_mid, veg_short, ha='right', va='center',
                 fontsize=FS_TICK, fontweight='bold', color=veg_col,
                 bbox=dict(boxstyle='round,pad='+str(PAD_VEG), fc='white', ec=veg_col, lw=BLW_VEG), zorder=5)

    # Veg -> junction0
    ax_tree.plot([VEG_R, J0_X], [veg_mid, veg_mid], color=veg_col, lw=LW_ROOT, zorder=1)

    lcz_mids = []
    for lcz_val, lc_groups in lcz_groups:
        lcz_col = '#444444'   # neutral: the LCZ_COLORS red/blue/gold collided with Built red and the PatchTST/LSTM tiles (2026-10-06)
        all_lcz_idx = []
        for lc_val, sta_idx in lc_groups:
            all_lcz_idx.extend(sta_idx)
        lcz_ys = y_pos[all_lcz_idx]
        lcz_mid = (lcz_ys.min() + lcz_ys.max()) / 2
        lcz_mids.append(lcz_mid)

        # Junction0 -> LCZ
        ax_tree.plot([J0_X, LCZ_R - LCZ_BW], [lcz_mid, lcz_mid], color=veg_col, lw=LW_BRANCH, zorder=1)

        # LCZ label
        ax_tree.text(LCZ_R - 0.15, lcz_mid, LCZ_SHORT[lcz_val], ha='right', va='center',
                     fontsize=FS_TICK, fontweight='bold', color=lcz_col,
                     bbox=dict(boxstyle='round,pad='+str(PAD_LCZ), fc='white', ec=lcz_col, lw=BLW_LCZ), zorder=5)

        # LCZ -> junction1
        ax_tree.plot([LCZ_R, J1_X], [lcz_mid, lcz_mid], color=lcz_col, lw=LW_BRANCH, zorder=1)

        lc_mids = []
        for lc_val, sta_idx in lc_groups:
            lc_col = LC_COLORS[lc_val]
            sta_ys = y_pos[sta_idx]
            lc_mid = (sta_ys.min() + sta_ys.max()) / 2
            lc_mids.append(lc_mid)

            # Junction1 -> LC
            ax_tree.plot([J1_X, LC_R - LC_BW], [lc_mid, lc_mid], color=lcz_col, lw=LW_LEAF, zorder=1)

            # LC label
            ax_tree.text(LC_R - 0.1, lc_mid, lc_val, ha='right', va='center',
                         fontsize=FS_DENSE, fontweight='bold', color=lc_col,
                         bbox=dict(boxstyle='round,pad='+str(PAD_LC), fc='white', ec=lc_col, lw=BLW_LC), zorder=5)

            # LC -> junction2
            ax_tree.plot([LC_R, J2_X], [lc_mid, lc_mid], color=lc_col, lw=LW_LEAF, zorder=1)

            # Junction2 vertical
            if len(sta_idx) > 1:
                ax_tree.plot([J2_X, J2_X], [sta_ys.min(), sta_ys.max()],
                             color=lc_col, lw=LW_LEAF, zorder=1)

            # Branches to stations
            for idx in sta_idx:
                ax_tree.plot([J2_X, STA_X - 0.3], [y_pos[idx], y_pos[idx]],
                             color=lc_col, lw=LW_LEAF, zorder=1)

        # Junction1 vertical
        if len(lc_mids) > 1:
            ax_tree.plot([J1_X, J1_X], [min(lc_mids), max(lc_mids)],
                         color=lcz_col, lw=LW_LEAF, zorder=1)

    # Junction0 vertical
    if len(lcz_mids) > 1:
        ax_tree.plot([J0_X, J0_X], [min(lcz_mids), max(lcz_mids)],
                     color=veg_col, lw=LW_BRANCH, zorder=1)

# Root node: "Urban morphology" connecting both surface cover groups
all_veg_mids = []
for veg, lcz_groups in full_tree:
    all_idx = []
    for lcz_val, lc_groups in lcz_groups:
        for lc_val, sta_idx in lc_groups:
            all_idx.extend(sta_idx)
    veg_ys = y_pos[all_idx]
    all_veg_mids.append((veg_ys.min() + veg_ys.max()) / 2)

ROOT_X = VEG_R - VEG_BW - 0.5
# Vertical line connecting surface cover groups
ax_tree.plot([ROOT_X, ROOT_X], [min(all_veg_mids), max(all_veg_mids)],
             color=COL_ROOT, lw=LW_ROOT, zorder=1)
# Horizontal lines from root to each surface cover group
for vm in all_veg_mids:
    ax_tree.plot([ROOT_X, VEG_R - VEG_BW], [vm, vm],
                 color=COL_ROOT, lw=LW_ROOT, zorder=1)
# Root label
root_mid = (min(all_veg_mids) + max(all_veg_mids)) / 2
ax_tree.text(ROOT_X, root_mid, 'Urban\nmorphology', ha='center', va='center',
             fontsize=FS_TICK, fontweight='bold', color=COL_ROOT,
             bbox=dict(boxstyle='round,pad='+str(PAD_VEG), fc='white', ec=COL_ROOT, lw=BLW_VEG), zorder=5)

# Test set sizes per station (for n= labels)
with open('ms_predictions_cache.pkl', 'rb') as _f:
    _cache = _pkl.load(_f)
_preds = _cache['ms_predictions']
TEST_N = {}
for sid, name, lc, lcz, sc in ORDERED_STATIONS_LCZ:
    _key = (str(sid), '5min', 'PatchTST')
    if _key in _preds:
        TEST_N[sid] = len(_preds[_key][0])
del _cache, _preds

# Station names + test set size
for i, sn in enumerate(station_labels):
    sid = ORDERED_STATIONS_LCZ[i][0]
    n_test = TEST_N.get(sid, 0)
    ax_tree.text(STA_X, y_pos[i] - 0.15, sn, ha='left', va='center', fontsize=FS_TICK)
    ax_tree.text(STA_X, y_pos[i] + 0.25, f'(n = {n_test:,})', ha='left', va='center',
                 fontsize=FS_DENSE_SM, color='#888888', fontstyle='italic')

ax_tree.axis('off')

# ---- TILE PANELS ----
def lighten(hex_color, factor=0.35):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1-r)*(1-factor), g + (1-g)*(1-factor), b + (1-b)*(1-factor))
CAND_LIGHT = {m: lighten(c) for m, c in CAND_COLORS.items()}

tile_axes = {}
for pidx, hz in enumerate(HORIZONS):
    if pidx == 0:
        ax = ax_tile0
    else:
        ax = fig.add_subplot(gs[0, pidx + 1], sharey=ax_tile0, sharex=ax_tile0)
    tile_axes[pidx] = ax

    ax.set_xlim(-0.5, n_mod - 0.5)
    tw, th = 0.68, 0.68
    for i in range(n_sta):
        for j in range(n_mod):
            x0 = j - tw / 2
            y0 = y_pos[i] - th / 2
            if sig_matrix[hz][i, j]:
                fc = CAND_COLORS[CANDIDATES[j]]
                ec = '#333333'; lw_r = 0.8
            elif beats_mae_matrix[hz][i, j]:
                fc = CAND_LIGHT[CANDIDATES[j]]
                ec = '#AAAAAA'; lw_r = 0.5
            else:
                fc = '#E8E8E8'; ec = '#CCCCCC'; lw_r = 0.5
            rect = FancyBboxPatch((x0, y0), tw, th, boxstyle='round,pad=0.02',
                                   facecolor=fc, edgecolor=ec, linewidth=lw_r, zorder=2)
            ax.add_patch(rect)
            if sig_matrix[hz][i, j]:
                val = skill_matrix[hz][i, j]
                if not np.isnan(val):
                    ax.text(j, y_pos[i], f'{val:+.0f}%', ha='center', va='center',
                            fontsize=FS_DENSE, color='white', fontweight='bold', zorder=3,
                            rotation=90)
            # pale tiles carry no mark: the old 'n.s.' was false for 17 of the 76 (significant against DLinear, failing only on
            # Persistence or on MAE); pale = beats DLinear, not a win (review 2026-10-03, item C6)

    ax.text(0.5, 1.05, HZ_LABELS[hz], transform=ax.transAxes,
            ha='center', va='bottom', fontsize=FS_LABEL, fontweight='bold', clip_on=False)

    # Model short labels at top of each column
    for j, cand in enumerate(CANDIDATES):
        ax.text(j, -0.5, MODEL_SHORT[cand], ha='center', va='bottom',
                fontsize=FS_TICK, color='black', clip_on=False)

    ax.set_yticks([]); ax.tick_params(axis='x', labelbottom=False, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

# --- Regime dendrogram above horizon labels ---
fig.canvas.draw()

REGIME_GROUPS = [
    ('Thermal inertia', [0, 1, 2]),
    ('Diurnal\ndecoupling', [3]),   # canonical regime-2 name (matches Fig 3 + body)
    ('Full diurnal\ncycle', [4]),
]

# Get figure-space x midpoints of each tile axis
hz_mids = []
for pidx in range(len(HORIZONS)):
    bbox = tile_axes[pidx].get_position()
    hz_mids.append((bbox.x0 + bbox.x1) / 2)

y_base = tile_axes[0].get_position().y1
LW_TREE = LW_BRANCH
TREE_COL = COL_ROOT

# Level 1: regime lines + labels
y_regime = y_base + 0.06
y_hz_top = y_base + 0.045
regime_mids = []
_regime_labels = []

for regime_label, hz_indices in REGIME_GROUPS:
    mids = [hz_mids[i] for i in hz_indices]
    x_left = min(mids)
    x_right = max(mids)
    x_mid = (x_left + x_right) / 2
    regime_mids.append(x_mid)

    for m in mids:
        fig.add_artist(plt.Line2D([m, m], [y_hz_top, y_regime],
                       transform=fig.transFigure, color=TREE_COL,
                       lw=LW_TREE, clip_on=False))
    if len(mids) > 1:
        fig.add_artist(plt.Line2D([x_left, x_right], [y_regime, y_regime],
                       transform=fig.transFigure, color=TREE_COL,
                       lw=LW_TREE, clip_on=False))
    _regime_labels.append((x_mid, regime_label))

# Level 2: root connecting all regimes
y_root = y_regime + 0.04
x_root_left = min(regime_mids)
x_root_right = max(regime_mids)

for rm in regime_mids:
    fig.add_artist(plt.Line2D([rm, rm], [y_regime + 0.002, y_root],
                   transform=fig.transFigure, color=TREE_COL,
                   lw=LW_TREE, clip_on=False))
fig.add_artist(plt.Line2D([x_root_left, x_root_right], [y_root, y_root],
               transform=fig.transFigure, color=TREE_COL,
               lw=LW_TREE, clip_on=False))

# Draw labels on top (after all lines)
for x_mid, regime_label in _regime_labels:
    fig.text(x_mid, y_regime + 0.005, regime_label,
             ha='center', va='bottom', fontsize=FS_LEGEND, fontstyle='italic',
             color=COL_ROOT, transform=fig.transFigure,
             bbox=dict(boxstyle='square,pad=0.08', fc='white', ec='none'))
# Root label
fig.text((x_root_left + x_root_right) / 2, y_root + 0.005, 'Forecast horizon',
         ha='center', va='bottom', fontsize=FS_LABEL, fontweight='bold',
         color=COL_ROOT, transform=fig.transFigure)

# ---- RIGHT SUMMARY 1: Per-station bars ----
ax_right = fig.add_subplot(gs[0, 6], sharey=ax_tile0)

station_sum_skill = np.zeros(n_sta)
station_n_sig = np.zeros(n_sta, dtype=int)
for hz in HORIZONS:
    for i in range(n_sta):
        for j in range(n_mod):
            if sig_matrix[hz][i, j]:
                station_sum_skill[i] += skill_matrix[hz][i, j]
                station_n_sig[i] += 1

max_skill = max(station_sum_skill.max(), 1)
for i in range(n_sta):
    lc = lc_list[i]
    bar_col = LC_COLORS[lc]
    if station_sum_skill[i] > 0:
        ax_right.barh(y_pos[i], station_sum_skill[i], height=0.6,
                      color=bar_col, edgecolor='#555', linewidth=0.4, alpha=0.85, zorder=3)
        ax_right.text(station_sum_skill[i] + 1, y_pos[i],
                      f'+{station_sum_skill[i]:.0f}%\nn={station_n_sig[i]}',   # 2026-10-13: same compact form as the bottom bars
                      ha='left', va='center', fontsize=FS_TICK, color='#333')

ax_right.set_xlim(0, max_skill * 1.35)
ax_right.set_xlabel('\u03a3 Skill (%)', fontsize=FS_LABEL)
ax_right.tick_params(axis='x', labelsize=FS_TICK)
ax_right.tick_params(axis='y', labelleft=False, length=0)
ax_right.spines['top'].set_visible(False)    # open style (match Fig 3, 2026-06-11)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.xaxis.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)
ax_right.set_axisbelow(True)


# ---- BOTTOM BARS ----
ax_bar0 = None
for pidx, hz in enumerate(HORIZONS):
    if pidx == 0:
        ax_bar = fig.add_subplot(gs[1, pidx + 1], sharex=tile_axes[pidx])
        ax_bar0 = ax_bar
    else:
        ax_bar = fig.add_subplot(gs[1, pidx + 1], sharex=tile_axes[pidx], sharey=ax_bar0)
    x_pos_bar = np.arange(n_mod)
    counts = [sig_summary[hz][m] for m in CANDIDATES]
    colors = [CAND_COLORS[m] for m in CANDIDATES]
    sum_pcts = []
    for j_bar in range(n_mod):
        model_skills = skill_matrix[hz][:, j_bar][sig_matrix[hz][:, j_bar]]
        sum_pcts.append(np.nansum(model_skills) if len(model_skills) > 0 else 0)
    bars = ax_bar.bar(x_pos_bar, sum_pcts, color=colors, width=0.6,
                      edgecolor='black', linewidth=0.5)
    prev_n_y = None
    for j_bar, (bar, val, spct) in enumerate(zip(bars, counts, sum_pcts)):
        if val > 0:
            cx_bar = bar.get_x() + bar.get_width() / 2
            short = bar.get_height() < 14   # bar too short for in-bar rotated text
            if short:
                ax_bar.text(cx_bar, bar.get_height() + 0.8, f'+{spct:.0f}%',
                            ha='center', va='bottom', fontsize=FS_DENSE_SM,
                            color='black', fontweight='bold', rotation=90)
                n_y = bar.get_height() + 12.5
            else:
                ax_bar.text(cx_bar, bar.get_height() / 2, f'+{spct:.0f}%',
                            ha='center', va='center', fontsize=FS_DENSE_SM,
                            color='white', fontweight='bold', rotation=90)
                n_y = bar.get_height() + 0.5
            # dodge only when the neighbour's label would sit at a similar height
            if prev_n_y is not None and abs(n_y - prev_n_y) < 5.0:
                n_y = prev_n_y + 5.0
            prev_n_y = n_y
            ax_bar.text(cx_bar + (0.10 if j_bar == 0 else 0.0), n_y, f'n={val}',   # 2026-10-13: compact; the first bar's label sits 0.10 right (centred between the axis line and the next bar)
                        ha='center', va='bottom', fontsize=FS_TICK, fontweight='bold')
        else:
            prev_n_y = None   # empty slot breaks adjacency
    ax_bar.set_ylim(0, 50)
    ax_bar.set_xlim(-0.5, n_mod - 0.5)
    ax_bar.set_xticks(x_pos_bar)
    ax_bar.set_xticklabels([])
    ax_bar.tick_params(axis='x', length=0)
    ax_bar.set_yticks([0, 25, 50])
    ax_bar.tick_params(axis='y', labelsize=FS_TICK)
    ax_bar.spines['top'].set_visible(False)      # open style (match Fig 3, 2026-06-11)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.axhline(25, color='#ccc', linestyle=':', linewidth=0.5, zorder=0)
    ax_bar.axhline(50, color='#ccc', linestyle=':', linewidth=0.5, zorder=0)
    if pidx == 0:
        ax_bar.set_ylabel('\u03a3 Skill (%)\nvs DLinear', fontsize=FS_LABEL,
                          rotation=0, labelpad=24, va='center')
    else:
        ax_bar.tick_params(axis='y', labelleft=False)


# ---- BOTTOM ROW 2: Stacked win-count bars (sig + ns) ----
ax_ns0 = None
for pidx, hz in enumerate(HORIZONS):
    if pidx == 0:
        ax_ns = fig.add_subplot(gs[2, pidx + 1], sharex=tile_axes[pidx], sharey=ax_ns0)
        ax_ns0 = ax_ns
    else:
        ax_ns = fig.add_subplot(gs[2, pidx + 1], sharex=tile_axes[pidx])
    x_pos_ns = np.arange(n_mod)
    sig_counts = [sig_summary[hz][m] for m in CANDIDATES]
    ns_counts = [ns_summary[hz][m] for m in CANDIDATES]
    colors_dark = [CAND_COLORS[m] for m in CANDIDATES]
    colors_light = [lighten(CAND_COLORS[m], factor=0.55) for m in CANDIDATES]
    # Sig portion (dark, bottom)
    bars_sig = ax_ns.bar(x_pos_ns, sig_counts, color=colors_dark, width=0.6,
                         edgecolor='black', linewidth=0.5, label='Sig.')
    # NS portion (light, stacked on top)
    bars_ns = ax_ns.bar(x_pos_ns, ns_counts, bottom=sig_counts,
                        color=colors_light, width=0.6,
                        edgecolor='#999', linewidth=0.4, label='no win')   # internal label, never drawn: the light bars count every pale tile, 17 of which are significant vs DLinear (2026-10-13)
    # Total count label on top
    for j_ns in range(n_mod):
        total = sig_counts[j_ns] + ns_counts[j_ns]
        if total > 0:
            ax_ns.text(x_pos_ns[j_ns], total + 0.3, f'{total}',
                       ha='center', va='bottom', fontsize=FS_TICK, fontweight='bold')
    ax_ns.set_ylim(0, 12)
    ax_ns.set_xlim(-0.5, n_mod - 0.5)
    ax_ns.set_xticks(x_pos_ns)
    ax_ns.set_xticklabels([MODEL_SHORT[m] for m in CANDIDATES], fontsize=FS_TICK)
    ax_ns.set_yticks([0, 5, 10])
    ax_ns.tick_params(axis='y', labelsize=FS_TICK)
    ax_ns.spines['top'].set_visible(False)       # open style (match Fig 3, 2026-06-11)
    ax_ns.spines['right'].set_visible(False)
    ax_ns.axhline(10, color='#ccc', linestyle=':', linewidth=0.5, zorder=0)
    if pidx == 0:
        ax_ns.set_ylabel('Beats DLinear\n(/10 sta.)', fontsize=FS_LABEL,   # was 'Wins': these bars count stations that beat DLinear on MAE; a "win" in the paper also needs significance against both benchmarks (figure check 2026-10-02)
                          rotation=0, labelpad=24, va='center')
    else:
        ax_ns.tick_params(axis='y', labelleft=False)


# ---- LEGEND in bottom-left (spans rows 1-2) ----
ax_leg = fig.add_subplot(gs[1:3, 0])
ax_leg.axis('off')
legend_elements = [
    Patch(fc=CAND_COLORS['PatchTST'], ec='#333', lw=0.8, label='PatchTST'),
    Patch(fc=CAND_COLORS['iTransformer'], ec='#333', lw=0.8, label='iTransformer'),
    Patch(fc=CAND_COLORS['LSTM'], ec='#333', lw=0.8, label='LSTM'),
    Patch(fc=CAND_COLORS['Attention-LSTM'], ec='#333', lw=0.8, label='Attn-LSTM'),
    Patch(fc='none', ec='#AAA', lw=0.8, label='Beats DLinear, no win'),   # drawn as four pale model tints by _HTints (2026-10-06): the tiles use CAND_LIGHT, not gray   # was 'Beats DLinear (n.s.)': 17 of the pale tiles are significant against DLinear (review 2026-10-03)
    Patch(fc='#E8E8E8', ec='#CCC', lw=0.8, label='Behind DLinear'),   # was 'DL >= DLinear', which could be read the wrong way (figure check 2026-10-02)
]
from matplotlib.legend_handler import HandlerPatch as _HP
from matplotlib.patches import Rectangle as _Rect
class _HTints(_HP):
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        out = []
        for i, m in enumerate(['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']):
            r = _Rect((xd + i * w / 4.0, yd), w / 4.0, h, fc=CAND_LIGHT[m], ec='none', transform=trans); out.append(r)
        out.append(_Rect((xd, yd), w, h, fc='none', ec='#AAA', lw=0.8, transform=trans))
        return out
_hmap = {legend_elements[4]: _HTints()}
ax_leg.legend(handles=legend_elements, handler_map=_hmap, loc='center',
              fontsize=FS_DENSE, frameon=True, fancybox=False, edgecolor='#cccccc',
              ncol=2, handletextpad=0.3, columnspacing=0.6, borderpad=0.5,
              title='Dual-benchmark win\n(Holm\u2013Bonferroni, p < 0.05)',   # the caption's term (2026-10-07)
              title_fontsize=FS_DENSE, labelspacing=0.4)
ax_leg.get_legend().get_frame().set_linewidth(0.4)   # figure-wide legend frame: #cccccc, 0.4 pt (2026-09-17)

# --- Dashed zoom boxes: Baker 6h and Canopied 6h -> Fig. 8 ---
hz_6h_idx = HORIZONS.index('6h')
ax_6h = tile_axes[hz_6h_idx]
baker_idx = 1    # Baker Building in ORDERED_STATIONS_LCZ
canopied_idx = 8  # Canopied Walkway

for sta_idx, box_color in [(baker_idx, '#8B4513'), (canopied_idx, '#2ca02c')]:
    box_x = -0.5 - 0.05
    box_y = y_pos[sta_idx] - 0.45
    box_w = n_mod + 0.1 - 0.02
    box_h = 0.9
    rect = plt.Rectangle((box_x, box_y), box_w, box_h,
                          facecolor='none', edgecolor=box_color,
                          linewidth=1.2, linestyle='--', zorder=10, clip_on=False)
    ax_6h.add_patch(rect)

from fig_config import savefig as _savefig
_savefig(plt, 'fig04_cross_station_landcover', bbox='fullwidth')  # exact FULL_W width for consistent print scale
plt.close()
