# -*- coding: utf-8 -*-
"""Teste de permutacao, 10 splits aleatorios e importancia das features."""
import datetime, json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from featurelib import load, build, SEED, TRAIN_END

rng = np.random.default_rng(SEED)
lines, dayvol, rec_unit = load()
train = [l for l in lines if l['date'] <= TRAIN_END]
test = [l for l in lines if l['date'] > TRAIN_END]
Xtr, Xte, names, _ = build(train, test, dayvol, rec_unit)
ytr = np.array([l['label'] for l in train]); yte = np.array([l['label'] for l in test])

iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr)
s = -iso.score_samples(Xte)
ap = average_precision_score(yte, s)
base = yte.mean()
print('PR-AUC observado: %.4f  | base rate %.4f  | lift %.2fx' % (ap, base, ap / base))

# --- 1. teste de permutacao: o lift distingue-se do acaso? ---------------
N = 2000
perm = np.array([average_precision_score(rng.permutation(yte), s) for _ in range(N)])
p_emp = (1 + (perm >= ap).sum()) / (N + 1)
print('\n1. TESTE DE PERMUTACAO (%d permutacoes do label)' % N)
print('   PR-AUC sob H0: media %.4f  p95 %.4f  max %.4f' % (perm.mean(), np.percentile(perm, 95), perm.max()))
print('   p-valor empirico: %.4f  -> %s' % (p_emp, 'DISTINGUE-SE do acaso' if p_emp < 0.05 else 'NAO se distingue do acaso'))

# --- 2. 10 splits aleatorios --------------------------------------------
print('\n2. DEZ SPLITS ALEATORIOS (mesma proporcao treino/teste)')
aps, lifts = [], []
allx = lines
for k in range(10):
    r2 = np.random.default_rng(SEED + k)
    # 0.665 = mesma proporcao treino/teste do hold-out temporal
    idx = r2.permutation(len(allx)); cut = int(len(allx) * 0.665)
    tr = [allx[i] for i in idx[:cut]]; te = [allx[i] for i in idx[cut:]]
    A, B, _, _ = build(tr, te, dayvol, rec_unit)
    yb = np.array([l['label'] for l in te])
    m = IsolationForest(n_estimators=300, random_state=SEED + k, n_jobs=-1).fit(A)
    sb = -m.score_samples(B)
    a = average_precision_score(yb, sb)
    aps.append(a); lifts.append(a / yb.mean())
print('   PR-AUC  media %.4f  ± %.4f   (min %.4f  max %.4f)' % (np.mean(aps), np.std(aps), min(aps), max(aps)))
print('   lift    media %.2fx ± %.2f' % (np.mean(lifts), np.std(lifts)))

# --- 3. importancia por permutacao das features -------------------------
print('\n3. IMPORTANCIA DAS FEATURES (queda no PR-AUC ao baralhar cada uma)')
imp = []
for j, nm in enumerate(names):
    d = []
    for _ in range(6):
        Z = Xte.copy(); Z[:, j] = rng.permutation(Z[:, j])
        d.append(ap - average_precision_score(yte, -iso.score_samples(Z)))
    imp.append((float(np.mean(d)), nm))
for v, nm in sorted(imp, reverse=True)[:8]:
    print('   %-22s %+.4f' % (nm, v))
json.dump(dict(ap=ap, base=float(base), p_perm=float(p_emp),
               splits_ap=aps, splits_lift=lifts,
               importance=sorted(imp, reverse=True)), open('robustness.json', 'w'), indent=1)
