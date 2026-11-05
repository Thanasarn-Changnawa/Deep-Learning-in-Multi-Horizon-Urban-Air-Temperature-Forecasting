# -*- coding: utf-8 -*-
"""Intervals for Fig 3 panels h and i (2026-09-24). Writes fig03_intervals.csv, read by build_fig03.py.

Per horizon, for the best candidate (lowest network-mean MAE among the four) and for DLinear:
  quantity 'gain_best'   : 100 * (1 - mean_stations MAE_best / mean_stations MAE_DLinear)   -> panel h whisker
  quantity 'mae_best'    : network-mean MAE of the best candidate (degC)                      (not drawn)
  quantity 'mae_dlinear' : network-mean MAE of DLinear (degC)
  quantity 'margin_best' : network-mean margin of the best candidate over the benchmark it beats by less (degC) -> panel i whisker
Two 95% constructions for each: A = exact station-level percentile bootstrap (all 92,378 station multisets with their
multinomial weights, no seed; the paper's interval, as Table S8) and TW = two-way, the same test days drawn with
replacement at every station (circular 1-day blocks; daily gains have lag-1 autocorrelation -0.08) and then one station
multiset drawn by its exact weight (seed 11, B = 3000). Predictions are mapped to calendar days with the NaN-aware
correlation alignment (feedback_alignment_method: overlap slice, 0.85 split, best offset; r must exceed 0.99).
"""
import io, sys, pickle, itertools, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, pandas as pd
CACHE = 'output/TF_for-upload_4(H100)__output__completed/output/run_2026-03-01_20-51/ms_predictions_cache.pkl'
OV0, OV1 = '2015-06-05', '2017-06-28'
C = ['PatchTST', 'iTransformer', 'LSTM', 'Attention-LSTM']; H = ['5min', '10min', '1h', '6h', '24h']
SEED, B = 11, 3000
ms = pickle.load(open(CACHE, 'rb'))['ms_predictions']; sids = sorted(set(k[0] for k in ms if k[1] == '6h')); n = len(sids)
TE = {}
for s in sids:
    df = pd.read_csv(f'00_dataset/5min/20 stations/{s}_5min_Interval.csv', parse_dates=['Datetime'], low_memory=False)
    df = df.dropna(subset=['Temp', 'RH', 'DewPt']).sort_values('Datetime').set_index('Datetime').loc[OV0:OV1]
    TE[s] = df.iloc[int(len(df) * 0.85):]
combos = list(itertools.combinations_with_replacement(range(n), n))
cnt = np.array([np.bincount(c, minlength=n) for c in combos])
W = np.array([math.factorial(n) / np.prod([math.factorial(int(k)) for k in c]) / n ** n for c in cnt])
def wq(v, w, p):
    o = np.argsort(v); cv = np.cumsum(w[o]); return float(v[o][np.searchsorted(cv, p)])
def daytable(hz, model):
    S, N, days_all = {}, {}, set()
    for s in sids:
        tdf = TE[s]; te = tdf['Temp'].values; p, act = ms[(s, hz, model)]; act = np.asarray(act).ravel(); m = len(act); chk = min(500, m)
        off = max(range(0, min(1000, len(te) - m) + 1), key=lambda o: np.corrcoef(act[:chk], te[o:o + chk])[0, 1])
        assert np.corrcoef(act[:chk], te[off:off + chk])[0, 1] > 0.99, (s, hz, model)
        e = pd.Series(np.abs(np.asarray(p).ravel() - act), index=tdf.index[off:off + m].normalize()); g = e.groupby(level=0)
        S[s] = g.sum(); N[s] = g.size(); days_all |= set(S[s].index)
    days = sorted(days_all)
    return (np.array([[S[s].get(d, 0.0) for d in days] for s in sids]), np.array([[N[s].get(d, 0) for d in days] for s in sids], float), len(days))
rows = []
for hz in H:
    net = {m: np.mean([np.abs(np.asarray(ms[(s, hz, m)][0]).ravel() - np.asarray(ms[(s, hz, m)][1]).ravel()).mean() for s in sids]) for m in C}
    best = min(net, key=net.get)
    Sb, Nb, D = daytable(hz, best); Sd, Nd, D2 = daytable(hz, 'DLinear'); Sp, Np, D3 = daytable(hz, 'Persistence'); assert D == D2 == D3
    MAEb, MAEd, MAEp = Sb.sum(1) / Nb.sum(1), Sd.sum(1) / Nd.sum(1), Sp.sum(1) / Np.sum(1)
    # exact station bootstrap
    stb, std_ = cnt @ MAEb / n, cnt @ MAEd / n; gA = 100 * (1 - (cnt @ MAEb) / (cnt @ MAEd))
    mA = np.minimum(cnt @ MAEd, cnt @ MAEp) / n - (cnt @ MAEb) / n      # margin over the binding benchmark, degC
    # two-way
    rng = np.random.default_rng(SEED); tw_b, tw_d, tw_g, tw_m = [], [], [], []
    for _ in range(B):
        w = np.bincount(rng.integers(0, D, size=D), minlength=D).astype(float)
        sb = (Sb @ w) / (Nb @ w); sd = (Sd @ w) / (Nd @ w); sp_ = (Sp @ w) / (Np @ w); c = cnt[rng.choice(len(W), p=W)]
        tw_b.append((c @ sb) / n); tw_d.append((c @ sd) / n); tw_g.append(100 * (1 - (c @ sb) / (c @ sd))); tw_m.append(min(c @ sd, c @ sp_) / n - (c @ sb) / n)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    rows.append(dict(horizon=hz, quantity='gain_best', model=best, point=100 * (1 - MAEb.mean() / MAEd.mean()), A_lo=wq(gA, W, .025), A_hi=wq(gA, W, .975), TW_lo=q(tw_g)[0], TW_hi=q(tw_g)[1]))
    rows.append(dict(horizon=hz, quantity='mae_best', model=best, point=MAEb.mean(), A_lo=wq(stb, W, .025), A_hi=wq(stb, W, .975), TW_lo=q(tw_b)[0], TW_hi=q(tw_b)[1]))
    rows.append(dict(horizon=hz, quantity='margin_best', model=best, point=min(MAEd.mean(), MAEp.mean()) - MAEb.mean(), A_lo=wq(mA, W, .025), A_hi=wq(mA, W, .975), TW_lo=q(tw_m)[0], TW_hi=q(tw_m)[1]))
    rows.append(dict(horizon=hz, quantity='mae_dlinear', model='DLinear', point=MAEd.mean(), A_lo=wq(std_, W, .025), A_hi=wq(std_, W, .975), TW_lo=q(tw_d)[0], TW_hi=q(tw_d)[1]))
    R = {r['quantity']: r for r in rows if r['horizon'] == hz}
    print('%-5s best %-14s gain %+5.1f [A %+5.1f, %+5.1f | TW %+5.1f, %+5.1f] | margin %.3f [%.3f, %.3f | %.3f, %.3f] | MAE best %.3f | DLinear %.3f' % (
        hz, best, R['gain_best']['point'], R['gain_best']['A_lo'], R['gain_best']['A_hi'], R['gain_best']['TW_lo'], R['gain_best']['TW_hi'],
        R['margin_best']['point'], R['margin_best']['A_lo'], R['margin_best']['A_hi'], R['margin_best']['TW_lo'], R['margin_best']['TW_hi'], R['mae_best']['point'], R['mae_dlinear']['point']))
pd.DataFrame(rows).to_csv('fig03_intervals.csv', index=False, float_format='%.6f'); print('written fig03_intervals.csv (%d rows)' % len(rows))
