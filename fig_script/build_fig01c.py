"""The four-stage validation protocol, in two deliberately different variants.

  supp : Elsevier supplement figure (Figure S5). FULL_W wide, portrait, full 10-fold LOSO
         grid and the hierarchy summary. Every label >= 5 pt (feedback_text_sizes).
  slide: 16:9 talk slide. Four columns, one per stage, at projection-legible type
         (>= 14 pt). It carries the FLOW and the headline numbers only.

Why they differ: the full protocol needs ~28 vertical text rows. Slide-legible type on a
13.33 in canvas is ~18 pt, which allows roughly 30 characters per column. The two cannot be
the same drawing. Every fact on the slide is a subset of the supplement's, taken from the
same FACTS block below, so the two cannot state different things.

  python fig_script/build_fig01c.py           -> both
  python fig_script/build_fig01c.py supp      -> one

Content audited against the code and manuscript 2026-08-26 (memo 9bt); corrections made then:
  - the generalization gap is LOSO minus PER-STATION MAE (S3.4.3, Table 3), not minus pooled
  - the 15 DM comparisons are per STATION-horizon family (S3.4.4 and supplement S3), not per horizon
  - the station is "Curran Parking" ("Curran Deck" occurs 0 times in the manuscript)
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig_config import FULL_W, savefig

C_STAGE = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
C_BG    = ['#E8F4FD', '#FFF3E0', '#E8F5E9', '#FFEBEE']
C_DATA  = {'train': '#4CAF50', 'val': '#FFC107', 'test': '#F44336'}

TITLE    = 'Four-Stage Validation Protocol'
STATIONS = ['10th St Bridge', '5th St Bridge', 'Canopied Walk', 'Hemphill Pkg', 'Baker Bldg',
            'Curran Parking', 'Wetlands', 'Grass Walk', 'Tech Square', 'Brittain Dr']
COVERS   = ['Exposed', 'Exposed', 'Tree canopy', 'Impervious', 'Built',
            'Impervious', 'Open green', 'Open green', 'Built', 'Tree canopy']

# Single source of truth. 'head' is the stage name; 'slide' is the projection-legible subset
# of 'full'. Nothing may appear on the slide that is not also true in the supplement.
FACTS = [
    dict(head='Stage 1: Hyperparameter Optimization', short='Stage 1\nTune',
         full=['Single station (10th St Bridge)  |  Single horizon (t+1 = 5 min)'
               '  |  Optuna TPE + Grid Search'],
         slide=['1 station, 1 horizon', 'Optuna TPE, then', 'learning rate × dropout grid',
                'Selected on validation MSE']),
    dict(head='Stage 2: Per-Station Training & Evaluation', short='Stage 2\nTrain',
         full=['10 stations × 5 horizons × (4 candidates + DLinear) = 250 independent runs'],
         ital=['Early stopping on the last 10% of the 85% fit block, i.e. 8.5% of the series (patience = 10)',
               'Test: MAE, RMSE, R² on held-out test set → mean ± SD across 10 stations'],
         slide=['10 stations × 5 horizons', '× 5 trainable models', '= 250 independent runs',
                'Split 70 / 15 / 15, chronological']),
    dict(head='Stage 3: Pooled Training & LOSO Cross-Validation', short='Stage 3\nGeneralize',
         slide=['Pooled: all 10 stations', 'trained together', 'LOSO: 10 folds, one',
                'station held out each time']),
    dict(head='Stage 4: Statistical Significance Testing', short='Stage 4\nTest',
         full=['Diebold–Mariano pairwise tests on test residuals  |  '
               '15 comparisons per station–horizon  |  Holm–Bonferroni correction'],
         ital=['Question answered: is model A significantly better than B, or only numerically?'],
         slide=['Diebold–Mariano pairwise', 'tests on test residuals',
                '15 per station–horizon', 'Holm–Bonferroni corrected']),
]
POOLED_LINE = 'train on all 10 stations concatenated → test on each station individually'
GAP_LINE    = ('Generalization gap = LOSO MAE − per-station MAE '
               '→ cost of deploying to unseen location')
LEGEND      = [('Training data', 'train'), ('Validation (early stopping)', 'val'),
               ('Test (metrics reported)', 'test')]


# ===================================================================== supplement
def build_supp():
    FS = dict(title=11, head=8.5, body=6.6, small=6.0, ital=6.0, fold=5.0, cell=5.0,
              cover=5.0, bar=5.0, sum_h=7.0, sum_a=6.6, sum_b=6.0, leg=6.2)
    PAD, STEP = 0.46, 0.235      # 0.46 minus two 0.10 box pads = 0.26 units of visible air
    BOXPAD = 0.10               # FancyBboxPatch grows each box this far beyond its band
    fig, ax = plt.subplots(figsize=(FULL_W, 10))
    ax.set_xlim(0, 10); ax.axis('off')
    cur = [0.0]
    def take(h):
        top = cur[0]; cur[0] -= h; return top
    def box(top, h, i):
        ax.add_patch(FancyBboxPatch((0.3, top - h), 9.4, h, boxstyle='round,pad=0.1',
                                    facecolor=C_BG[i], edgecolor=C_STAGE[i], linewidth=1.5))
    def arrow(y, label=None):
        y -= BOXPAD                                   # clear the box outline first
        ax.annotate('', xy=(5, y - 0.25), xytext=(5, y - 0.02),
                    arrowprops=dict(arrowstyle='-|>', color='#555', lw=1.1, mutation_scale=9))
        if label:
            ax.text(5.18, y - 0.12, label, fontsize=FS['small'], fontstyle='italic',
                    color='#555', va='center')
    def databar(y, segs):
        left = 5.5
        for frac, key, lab, col in segs:
            ax.barh(y, 4.0 * frac, left=left, height=0.13, color=C_DATA[key], edgecolor='white')
            ax.text(left + 4.0 * frac / 2, y, lab, fontsize=FS['bar'], ha='center',
                    va='center', color=col, fontweight='bold')
            left += 4.0 * frac
        ax.text(5.35, y, 'Data:', fontsize=FS['bar'], ha='right', va='center')

    # 2026-10-12: no title drawn on the figure (Urban Climate guide: a caption's title is "not displayed on the
    # figure itself"); only enough top air for the Stage 1 box outline. The slide variant keeps its title.
    cur[0] -= BOXPAD + 0.02

    # Stage 1
    top = take(1.00); box(top, 1.00, 0)
    ax.text(0.5, top - 0.22, FACTS[0]['head'], fontsize=FS['head'], fontweight='bold',
            color=C_STAGE[0], va='center')
    ax.text(0.5, top - 0.55, FACTS[0]['full'][0], fontsize=FS['body'], va='center', color='#333')
    databar(top - 0.82, [(0.70, 'train', '70%', 'white'), (0.15, 'val', '15%', 'black')])   # same scale as Stage 2; the test 15% is not used in tuning
    for xx, nm in ((5.5 + 4.0 * 0.70 / 2, 'Training'), (5.5 + 4.0 * 0.70 + 4.0 * 0.15 / 2, 'Validation')):   # 2026-10-13: names above, as in Stage 2
        ax.text(xx, top - 0.82 + 0.09, nm, fontsize=FS['bar'], ha='center', va='bottom', fontweight='bold', color='#333')
    arrow(cur[0], 'configs locked'); cur[0] -= PAD

    # Stage 2
    top = take(1.55); box(top, 1.55, 1)
    ax.text(0.5, top - 0.22, FACTS[1]['head'], fontsize=FS['head'], fontweight='bold',
            color=C_STAGE[1], va='center')
    ax.text(0.5, top - 0.55, FACTS[1]['full'][0], fontsize=FS['body'], va='center', color='#333')
    databar(top - 0.82, [(0.765, 'train', '76.5%', 'white'), (0.085, 'val', '8.5%', 'black'),
                         (0.15, 'test', '15%', 'white')])   # as the code splits it: 85% fit block, last 10% of it = early-stopping monitor (review 2026-09-22)
    # 2026-10-13 (the author): shares inside the bar, the legend's names above it; the narrow early-stopping
    # segment's name sits one row higher with a leader line so it clears 'Test'
    yb = top - 0.82; x0b = 5.5
    xs = [x0b + 4.0 * 0.765 / 2, x0b + 4.0 * 0.765 + 4.0 * 0.085 / 2, x0b + 4.0 * 0.85 + 4.0 * 0.15 / 2]
    for xx, nm in ((xs[0], 'Training'), (xs[2], 'Test')):
        ax.text(xx, yb + 0.09, nm, fontsize=FS['bar'], ha='center', va='bottom', fontweight='bold', color='#333')
    ax.annotate('Validation (early stopping)', xy=(xs[1], yb + 0.065), xytext=(xs[1], yb + 0.22), ha='center', va='bottom',
                fontsize=FS['bar'], fontweight='bold', color='#333', arrowprops=dict(arrowstyle='-', lw=0.6, color='#555', shrinkA=0.5, shrinkB=0))
    for k, s in enumerate(FACTS[1]['ital']):
        ax.text(0.5, top - 1.08 - k * 0.26, s, fontsize=FS['ital'], va='center',
                color='#555', fontstyle='italic')
    arrow(cur[0]); cur[0] -= PAD

    # Stage 3
    h3 = 0.95 + 10 * STEP + 0.45
    top = take(h3); box(top, h3, 2)
    ax.text(0.5, top - 0.22, FACTS[2]['head'], fontsize=FS['head'], fontweight='bold',
            color=C_STAGE[2], va='center')
    ax.text(0.5, top - 0.53, 'Pooled:', fontsize=FS['body'], fontweight='bold',
            va='center', color='#333')
    ax.text(1.62, top - 0.53, POOLED_LINE, fontsize=FS['body'], va='center', color='#333')
    ax.text(0.5, top - 0.80, 'LOSO (10 folds):', fontsize=FS['body'], fontweight='bold',
            va='center', color='#333')
    g = top - 1.05
    cx = lambda s: 1.85 + s * 0.775
    for s in range(10):
        ax.text(cx(s) + 0.30, g + 0.15, COVERS[s], fontsize=FS['cover'], ha='center',
                va='center', color='#555', fontstyle='italic')
    for r in range(10):
        fy = g - r * STEP
        ax.text(0.5, fy, 'Fold %d: %s' % (r + 1, STATIONS[r]), fontsize=FS['fold'],
                va='center', color='#555', fontweight='bold')
        for s in range(10):
            held = s == r
            ax.add_patch(FancyBboxPatch((cx(s) - 0.05, fy - 0.085), 0.70, 0.17,
                                        boxstyle='round,pad=0.02',
                                        # 2026-09-30: the last-stacked training station (notebook cell 92 stacks in STATIONS order and
                                        # early-stops on the last 10% of the stacked sequences) is the monitor, in the validation colour
                                        facecolor=C_DATA['test' if held else ('val' if s == (8 if r == 9 else 9) else 'train')],
                                        edgecolor='none', alpha=0.8 if held else 0.6))
            if held:
                ax.text(cx(s) + 0.30, fy, 'TEST', fontsize=FS['cell'], ha='center',
                        va='center', color='white', fontweight='bold')
    ax.text(0.5, g - 10 * STEP - 0.04, GAP_LINE, fontsize=FS['ital'], va='center',
            color='#555', fontstyle='italic')
    arrow(cur[0]); cur[0] -= PAD

    # Stage 4
    top = take(1.05); box(top, 1.05, 3)
    ax.text(0.5, top - 0.22, FACTS[3]['head'], fontsize=FS['head'], fontweight='bold',
            color=C_STAGE[3], va='center')
    ax.text(0.5, top - 0.56, FACTS[3]['full'][0], fontsize=FS['body'], va='center', color='#333')
    ax.text(0.5, top - 0.85, FACTS[3]['ital'][0], fontsize=FS['ital'], va='center',
            color='#555', fontstyle='italic')
    cur[0] -= PAD + 0.10

    # hierarchy summary
    rule = cur[0]
    ax.plot([0.5, 9.5], [rule, rule], color='#ccc', linewidth=0.5)
    ax.text(5, rule - 0.22, 'Validation Hierarchy Summary', fontsize=FS['sum_h'],
            fontweight='bold', ha='center', va='center', color='#333')
    s_top = rule - 0.42
    rows = [('Stage 1', 'Config selection', 'internal validation at 1 station'),
            ('Stage 2', 'Performance estimation', '10 per-station test sets'),   # 17 Oct 2026: the stations share most test days (supplement S3), so not 'independent'
            ('Stage 3', 'Spatial generalization', 'LOSO: works at unseen sites?'),
            ('Stage 4', 'Statistical rigor', 'DM tests: beyond noise?')]
    for i, (st, purpose, method) in enumerate(rows):
        x0 = 0.42 + i * 2.34
        ax.add_patch(FancyBboxPatch((x0, s_top - 0.72), 2.16, 0.72, boxstyle='round,pad=0.05',
                                    facecolor=C_STAGE[i], edgecolor='none', alpha=0.15))
        c = x0 + 1.08
        ax.text(c, s_top - 0.19, st, fontsize=FS['sum_a'], fontweight='bold', ha='center',
                va='center', color=C_STAGE[i])
        ax.text(c, s_top - 0.40, purpose, fontsize=FS['sum_b'], ha='center', va='center', color='#333')
        ax.text(c, s_top - 0.59, method, fontsize=FS['sum_b'] - 0.5, ha='center', va='center',
                color='#666', fontstyle='italic')
    cur[0] = s_top - 0.72 - PAD

    leg = cur[0] - 0.12
    for i, (lab, key) in enumerate(LEGEND):
        x = 1.35 + i * 3.05
        ax.add_patch(FancyBboxPatch((x, leg - 0.09), 0.32, 0.18, boxstyle='round,pad=0.02',
                                    facecolor=C_DATA[key], edgecolor='none', alpha=0.75))
        ax.text(x + 0.48, leg, lab, fontsize=FS['leg'], va='center', color='#333')
    return fig, ax, leg - 0.30, FULL_W


# ========================================================================== slide
def build_slide():
    W = 13.33
    FS = dict(title=26, head=18, body=14, note=12.5, leg=13)
    fig, ax = plt.subplots(figsize=(W, 7))
    ax.set_xlim(0, 10); ax.axis('off')

    ax.text(5, -0.30, TITLE, fontsize=FS['title'], fontweight='bold', ha='center', va='center')

    col_w, gutter = 2.12, 0.31          # 4*2.12 + 3*0.31 = 9.41, leaving a real side margin
    x0 = (10 - (4 * col_w + 3 * gutter)) / 2
    top, height = -0.85, 2.16           # header 0.62 + four 0.34 rows + breathing room
    for i, f in enumerate(FACTS):
        x = x0 + i * (col_w + gutter)
        ax.add_patch(FancyBboxPatch((x, top - height), col_w, height, boxstyle='round,pad=0.06',
                                    facecolor=C_BG[i], edgecolor=C_STAGE[i], linewidth=2.0))
        ax.add_patch(FancyBboxPatch((x, top - 0.62), col_w, 0.62, boxstyle='round,pad=0.06',
                                    facecolor=C_STAGE[i], edgecolor='none'))
        name, verb = f['short'].split('\n')
        ax.text(x + col_w / 2, top - 0.24, name, fontsize=FS['body'], fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(x + col_w / 2, top - 0.47, verb, fontsize=FS['head'], fontweight='bold',
                ha='center', va='center', color='white')
        for k, line in enumerate(f['slide']):
            ax.text(x + col_w / 2, top - 0.92 - k * 0.34, line, fontsize=FS['note'],
                    ha='center', va='center', color='#333')
        if i < 3:                                     # flow arrow into the next column
            ax.annotate('', xy=(x + col_w + gutter - 0.03, top - height / 2),
                        xytext=(x + col_w + 0.03, top - height / 2),
                        arrowprops=dict(arrowstyle='-|>', color='#666', lw=2.2, mutation_scale=18))

    # No colour legend here: the slide draws no data bars and no fold grid, so the
    # train/val/test swatches would describe nothing on this canvas.
    y = top - height - 0.40
    ax.text(5, y, GAP_LINE, fontsize=FS['note'], ha='center', va='center',
            color='#555', fontstyle='italic')
    return fig, ax, y - 0.30, W


def emit(variant):
    fig, ax, bottom, w = (build_supp() if variant == 'supp' else build_slide())
    ax.set_ylim(bottom, 0.06)
    span = 0.06 - bottom
    h = w * span / 10.0
    fig.set_size_inches(w, h)
    ax.set_position([0, 0, 1, 1])
    name = 'fig01c_validation_protocol' + ('' if variant == 'supp' else '_slide')
    savefig(fig, name)
    plt.close(fig)
    fits = '' if variant == 'supp' else ('   16:9 fit: %s' % ('YES' if h <= 7.5 else 'NO'))
    print('  %-5s -> %-40s %.2f x %.2f in%s' % (variant, name, w, h, fits))


if __name__ == '__main__':
    for v in (sys.argv[1:] or ['supp', 'slide']):
        emit(v)
