# -*- coding: utf-8 -*-
"""Avaliacao NAO-AUMENTADA de RQ1 sobre o log real de discrepancias.

Ver build_dataset.py para a construcao da populacao. Aqui:
  - features calculaveis ANTES da correcao (a quantidade corrigida do PHC
    nunca entra, nem nenhuma funcao dela);
  - hold-out temporal identico ao do Capitulo 4 (treino 18/05-14/06,
    teste 15/06-28/06);
  - Isolation Forest e autoencoder, nao supervisionados;
  - PR-AUC como metrica principal + precision/recall no limiar da taxa
    de anomalia observada no treino.
"""
import json, csv, datetime, math, os
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# Diretoria com fichas_tecnicas_flat.csv (produzido por data-crosscheck/).
DATA_DIR = os.environ.get('DATA_DIR', './data')

SEED = 42
TRAIN_END = datetime.date(2026, 6, 14)
rng = np.random.default_rng(SEED)

lines = json.load(open('lines.json'))
for l in lines:
    l['date'] = datetime.date.fromisoformat(l['date'])
dayvol = {tuple(k.split('|')): v for k, v in json.load(open('dayvol.json')).items()}
dayvol = {(a, datetime.date.fromisoformat(b)): v for (a, b), v in dayvol.items()}

# custo unitario implicito da ficha tecnica, por ref ERP
rec_unit = defaultdict(list)
for r in csv.DictReader(open(os.path.join(DATA_DIR, 'fichas_tecnicas_flat.csv'))):
    try:
        q, pax, cp = float(r['qtd']), float(r['pax']), float(r['custo_pax'])
    except (ValueError, TypeError):
        continue
    if q > 0 and pax > 0 and cp > 0:
        rec_unit[r['erp_ref']].append(cp * pax / q)
rec_unit = {k: float(np.median(v)) for k, v in rec_unit.items()}
print('refs com custo unitario implicito na ficha: %d' % len(rec_unit))

train = [l for l in lines if l['date'] <= TRAIN_END]
test = [l for l in lines if l['date'] > TRAIN_END]
print('treino %d linhas (%d pos) | teste %d linhas (%d pos)'
      % (len(train), sum(l['label'] for l in train), len(test), sum(l['label'] for l in test)))

# --- estatisticas por artigo, SO do treino -------------------------------
stats, freq = {}, Counter()
byref = defaultdict(list)
for l in train:
    byref[l['ref']].append(l['qty']); freq[l['ref']] += 1
for k, v in byref.items():
    stats[k] = (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0, len(v))

CATS = {}
def cat_index(name, val):
    d = CATS.setdefault(name, {})
    return d.setdefault(val, len(d))

def featurise(l):
    m, s, n = stats.get(l['ref'], (np.nan, np.nan, 0))
    z = (l['qty'] - m) / s if (s and s > 0) else np.nan
    vol_q, vol_v = dayvol.get((l['stream'], l['date']), (np.nan, np.nan))
    ru = rec_unit.get(l['ref'])
    cost_div = (l['unit_cost'] - ru) / ru if (ru and ru > 0 and l['unit_cost'] > 0) else np.nan
    return dict(
        qty=l['qty'],
        log_qty=math.log1p(max(l['qty'], 0)),
        value=l['qty'] * l['unit_cost'],
        unit_cost=l['unit_cost'],
        z_hist=z,
        abs_z_hist=abs(z) if z == z else np.nan,
        qty_share_day=l['qty'] / vol_q if (vol_q and vol_q > 0) else np.nan,
        art_freq=freq.get(l['ref'], 0),
        art_seen=1.0 if l['ref'] in stats else 0.0,
        cost_div_recipe=cost_div,
        dow=l['date'].weekday(),
        week=(l['date'] - datetime.date(2026, 5, 18)).days // 7,
        fam=cat_index('fam', l['ref'][:2]),
        wh=cat_index('wh', l['wh']),
        stream=cat_index('stream', l['stream']),
    )

FEATS = list(featurise(train[0]).keys())
def matrix(ls):
    return np.array([[featurise(l)[f] for f in FEATS] for l in ls], dtype=float)

Xtr_raw, Xte_raw = matrix(train), matrix(test)
ytr = np.array([l['label'] for l in train]); yte = np.array([l['label'] for l in test])

# imputacao pela mediana do treino + indicador de ausencia
med = np.nanmedian(Xtr_raw, axis=0)
def prep(X):
    miss = np.isnan(X).astype(float)
    Xf = np.where(np.isnan(X), med, X)
    keep = miss.sum(axis=0) > 0
    return np.hstack([Xf, miss[:, keep]])
miss_tr = np.isnan(Xtr_raw); keep = miss_tr.sum(axis=0) > 0
Xtr, Xte = prep(Xtr_raw), prep(Xte_raw)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
print('features: %d (%d base + %d indicadores de ausencia)' % (Xtr.shape[1], len(FEATS), keep.sum()))

# --- verificacao anti-leakage -------------------------------------------
print('\n--- correlacao de cada feature com o label (treino) ---')
cors = []
for i, f in enumerate(FEATS):
    x = Xtr_raw[:, i]; ok = ~np.isnan(x)
    c = np.corrcoef(x[ok], ytr[ok])[0, 1] if ok.sum() > 10 and np.std(x[ok]) > 0 else 0.0
    cors.append((abs(c), f, c))
for a, f, c in sorted(cors, reverse=True)[:6]:
    print('   %-18s r=%+.3f' % (f, c))
assert max(a for a, _, _ in cors) < 0.5, 'FEATURE COM CORRELACAO SUSPEITA — possivel leakage'
print('   (nenhuma acima de 0.5 — sem sinal de leakage)')

def at_rate(scores_tr, scores_te, y_te, rate):
    thr = np.quantile(scores_tr, 1 - rate)
    pred = (scores_te >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average='binary', zero_division=0)
    return p, r, f, pred.sum()

def boot_ci(y, s, thr, n=2000):
    idx = np.arange(len(y)); ps, rs = [], []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        pred = (s[b] >= thr).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y[b], pred, average='binary', zero_division=0)
        ps.append(p); rs.append(r)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(ps), q(rs)

RATE = ytr.mean()
results = {}

# --- Isolation Forest ----------------------------------------------------
iso = IsolationForest(n_estimators=300, max_samples='auto', random_state=SEED, n_jobs=-1).fit(Xtr)
s_tr, s_te = -iso.score_samples(Xtr), -iso.score_samples(Xte)
results['Isolation Forest'] = (s_tr, s_te)

# --- Autoencoder (MLP a reconstruir a entrada) ---------------------------
ae = MLPRegressor(hidden_layer_sizes=(16, 6, 16), activation='relu', solver='adam',
                  max_iter=600, random_state=SEED, early_stopping=True, n_iter_no_change=20)
ae.fit(Xtr, Xtr)
s_tr_ae = ((ae.predict(Xtr) - Xtr) ** 2).mean(axis=1)
s_te_ae = ((ae.predict(Xte) - Xte) ** 2).mean(axis=1)
results['Autoencoder'] = (s_tr_ae, s_te_ae)

print('\n' + '=' * 74)
print('AVALIACAO NAO-AUMENTADA — hold-out temporal (teste 15-28 Jun 2026)')
print('=' * 74)
print('base rate no teste: %.4f  (%d positivos em %d linhas)' % (yte.mean(), yte.sum(), len(yte)))
out = []
for name, (str_, ste) in results.items():
    ap = average_precision_score(yte, ste)
    thr = np.quantile(str_, 1 - RATE)
    p, r, f, nflag = at_rate(str_, ste, yte, RATE)
    (pl, ph), (rl, rh) = boot_ci(yte, ste, thr)
    lift = ap / yte.mean()
    print('\n%s' % name)
    print('  PR-AUC        %.4f   (base rate %.4f  ->  lift %.2fx)' % (ap, yte.mean(), lift))
    print('  precision     %.4f   IC95%% [%.3f, %.3f]' % (p, pl, ph))
    print('  recall        %.4f   IC95%% [%.3f, %.3f]' % (r, rl, rh))
    print('  F1            %.4f   (%d linhas sinalizadas)' % (f, nflag))
    out.append(dict(model=name, prauc=ap, base=float(yte.mean()), lift=lift,
                    precision=p, recall=r, f1=f, flagged=int(nflag),
                    p_ci=[pl, ph], r_ci=[rl, rh]))
    # por fluxo
    for st in sorted(set(l['stream'] for l in test)):
        m = np.array([l['stream'] == st for l in test])
        if yte[m].sum() == 0:
            print('    %-24s sem positivos no teste' % st); continue
        apf = average_precision_score(yte[m], ste[m])
        print('    %-24s n=%4d pos=%3d  PR-AUC %.4f  (base %.4f)'
              % (st, m.sum(), yte[m].sum(), apf, yte[m].mean()))
json.dump(out, open('results_nonaugmented.json', 'w'), indent=1)
print('\ngravado results_nonaugmented.json')
