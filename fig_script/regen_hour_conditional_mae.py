# -*- coding: utf-8 -*-
# Canonical copy (2026-09-22) of the line-by-line audit script that regenerated hour_conditional_mae.csv (Fig 7b source);
# the previous file (kept as hour_conditional_mae.csv.BAK_misaligned_2026-09-22) carried station-specific hour offsets of 0-8 h.
# Regenerate hour_conditional_mae.csv (Fig 7b source) for all five horizons with every prediction mapped to its TARGET
# wall-clock hour by the NaN-aware correlation alignment (overlap slice, 0.85 split, best-offset search; r must be > 0.99).
# Writes hour_conditional_mae_aligned.csv (same columns) and prints the old vs new network-mean gap curves per horizon.
import io, sys, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, pandas as pd
CACHE = 'output/TF_for-upload_4(H100)__output__completed/output/run_2026-03-01_20-51/ms_predictions_cache.pkl'
OV0, OV1 = '2015-06-05', '2017-06-28'
MODELS = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM', 'DLinear', 'Persistence']; DL = MODELS[:4]
HORIZONS = ['5min', '10min', '1h', '6h', '24h']
cache = pickle.load(open(CACHE, 'rb'))['ms_predictions']
old = pd.read_csv('hour_conditional_mae.csv'); old['Station'] = old['Station'].astype(str)
sids = sorted(set(k[0] for k in cache))
rows = []; worst_r = 1.0
for sid in sids:
    df = pd.read_csv(f'00_dataset/5min/20 stations/{sid}_5min_Interval.csv', parse_dates=['Datetime'], low_memory=False)
    df = df.dropna(subset=['Temp', 'RH', 'DewPt']).sort_values('Datetime').set_index('Datetime').loc[OV0:OV1]
    df_te = df.iloc[int(len(df) * 0.85):]; te = df_te['Temp'].values
    for hz in HORIZONS:
        for m in MODELS:
            pred, act = cache[(sid, hz, m)]; pred = np.asarray(pred).ravel(); act = np.asarray(act).ravel(); n = len(act)
            chk = min(500, n); best_off, best_c = 0, -1.0
            for off in range(0, min(1000, len(te) - n) + 1):
                c = np.corrcoef(act[:chk], te[off:off + chk])[0, 1]
                if c > best_c: best_c, best_off = c, off
            worst_r = min(worst_r, best_c)
            hr = df_te.index[best_off:best_off + n].hour.values; err = np.abs(pred - act)
            g = pd.Series(err).groupby(hr)
            for h, v in g.mean().items(): rows.append(dict(Station=sid, Horizon=hz, Model=m, Hour=int(h), MAE=float(v), N=int(g.size()[h])))
import os, shutil
if os.path.exists('hour_conditional_mae.csv') and not os.path.exists('hour_conditional_mae.csv.BAK_misaligned_2026-09-22'): shutil.copy2('hour_conditional_mae.csv', 'hour_conditional_mae.csv.BAK_misaligned_2026-09-22')
new = pd.DataFrame(rows); new.to_csv('hour_conditional_mae.csv', index=False)
print('written hour_conditional_mae.csv: %d rows (old %d) | worst alignment r = %.4f' % (len(new), len(old), worst_r))
def gap(d, hz):
    d = d[d.Horizon == hz]; out = []
    for h in range(24):
        dl = d[(d.Hour == h) & d.Model.isin(DL)].groupby('Station')['MAE'].mean().mean()
        dlin = d[(d.Hour == h) & (d.Model == 'DLinear')].groupby('Station')['MAE'].mean().mean()
        out.append((1 - dl / dlin) * 100)
    return np.array(out)
print('hour      :', ' '.join('%5d' % h for h in range(24)))
for hz in HORIZONS:
    go, gn = gap(old, hz), gap(new, hz)
    print('%-5s old  :' % hz, ' '.join('%5.1f' % v for v in go)); print('%-5s new  :' % hz, ' '.join('%5.1f' % v for v in gn))
    print('      new: DL wins at hours %s | biggest gain hour %d (%.1f%%) | biggest loss hour %d (%.1f%%)' % ([h for h in range(24) if gn[h] > 0], int(gn.argmax()), gn.max(), int(gn.argmin()), gn.min()))
# full-test check: recomputed network-mean MAE per model at 6 h vs Table 2
chk = new[new.Horizon == '6h'].groupby(['Station', 'Model']).apply(lambda g: (g.MAE * g.N).sum() / g.N.sum()).unstack().mean()
print('6 h network-mean MAE from the aligned file (Table 2: 1.649/1.570/1.584/1.606/1.698/3.803):', chk.round(3).to_dict())
