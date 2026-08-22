# -*- coding: utf-8 -*-
"""Verificacao: o resultado do conjunto aumentado (Tabela 4.3) muda com um
hold-out TEMPORAL em vez do split aleatorio estratificado?

Tres variantes, tudo o resto igual ao evaluate_anomaly_detection_augmented.py
original (mesma feature unica, mesmo modelo, seed 42):
  A) split aleatorio estratificado 70/30            <- o que a tese usa
  B) hold-out temporal (treino <=14/06, teste 15-28/06)
  C) hold-out temporal + baseline do z-score recalculado SO no treino
"""
import numpy as np, pandas as pd, datetime, os
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

SEED = 42
# augmented_real_dataset.csv e produzido por build_augmented_real_dataset.py
# e nao e versionado (dados reais ao nivel da linha) -- ver .gitignore.
P = os.environ.get('AUGMENTED_CSV', './augmented_real_dataset.csv')
TRAIN_END = datetime.date(2026, 6, 14)

df = pd.read_csv(P)
df['deviation_zscore_historical'] = df['deviation_zscore_historical'].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['deviation_zscore_historical']).reset_index(drop=True)
df['dt'] = pd.to_datetime(df['Data'], format='%d.%m.%Y', errors='coerce').dt.date
print('linhas: %d | anomalias: %d (%.1f%%)' % (len(df), df['label'].sum(), 100*df['label'].mean()))
print('datas: %s -> %s | sem data: %d' % (df['dt'].min(), df['dt'].max(), df['dt'].isna().sum()))

def run(tr, te, feat='deviation_zscore_historical', tag=''):
    Xtr = tr[[feat]].values; Xte = te[[feat]].values
    ytr = tr['label'].values; yte = te['label'].values
    rate = ytr.mean()
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    iso = IsolationForest(n_estimators=200, contamination=rate, random_state=SEED).fit(A)
    pred = (iso.predict(B) == -1).astype(int)
    s = -iso.decision_function(B)
    p = precision_score(yte, pred, zero_division=0)
    r = recall_score(yte, pred, zero_division=0)
    f = f1_score(yte, pred, zero_division=0)
    ap = average_precision_score(yte, s)
    print('\n%s' % tag)
    print('  treino %d (%d anom, %.1f%%) | teste %d (%d anom, %.1f%%)'
          % (len(tr), ytr.sum(), 100*rate, len(te), yte.sum(), 100*yte.mean()))
    print('  precision %.3f   recall %.3f   F1 %.3f   PR-AUC %.3f  (lift %.2fx)'
          % (p, r, f, ap, ap/yte.mean()))
    return dict(tag=tag, precision=p, recall=r, f1=f, prauc=ap, lift=ap/yte.mean())

res = []
# A) o protocolo da tese
a, b = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=SEED)
res.append(run(a, b, tag='A) split aleatorio estratificado 70/30  [protocolo da Tabela 4.3]'))

# B) hold-out temporal
d = df.dropna(subset=['dt'])
a2 = d[d['dt'] <= TRAIN_END]; b2 = d[d['dt'] > TRAIN_END]
res.append(run(a2, b2, tag='B) hold-out temporal (treino <=14/06, teste 15-28/06)'))

# C) temporal + baseline do z-score so do treino
d2 = d.copy()
g = d2[d2['dt'] <= TRAIN_END].groupby('Ref')['Quant_original']
m, s2 = g.mean(), g.std()
d2['zs_train'] = (d2['Quant'] - d2['Ref'].map(m)) / d2['Ref'].map(s2)
d2['zs_train'] = d2['zs_train'].replace([np.inf, -np.inf], np.nan)
d2 = d2.dropna(subset=['zs_train'])
a3 = d2[d2['dt'] <= TRAIN_END]; b3 = d2[d2['dt'] > TRAIN_END]
res.append(run(a3, b3, feat='zs_train', tag='C) hold-out temporal + baseline do z-score so do treino'))

print('\n' + '='*72)
print('%-58s %8s %8s' % ('variante', 'precision', 'recall'))
for r in res:
    print('%-58s %8.3f %8.3f' % (r['tag'][:58], r['precision'], r['recall']))
print('\nreferencia da tese (Tabela 4.3, 10 splits aleatorios): 0.578 ± 0.029 / 0.576 ± 0.029')
pd.DataFrame(res).to_csv('check_augmented_temporal.csv', index=False)
