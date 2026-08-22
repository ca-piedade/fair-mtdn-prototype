# -*- coding: utf-8 -*-
"""Construcao de features para a avaliacao nao-aumentada. Partilhado pelo
script de avaliacao e pelo de robustez."""
import json, csv, datetime, math, os
import numpy as np
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler

# Diretoria com fichas_tecnicas_flat.csv (produzido por data-crosscheck/).
DATA_DIR = os.environ.get('DATA_DIR', './data')

SEED = 42
TRAIN_END = datetime.date(2026, 6, 14)
WINDOW_START = datetime.date(2026, 5, 18)


def load():
    lines = json.load(open('lines.json'))
    for l in lines:
        l['date'] = datetime.date.fromisoformat(l['date'])
    dv = json.load(open('dayvol.json'))
    dayvol = {}
    for k, v in dv.items():
        a, b = k.split('|')
        dayvol[(a, datetime.date.fromisoformat(b))] = v
    rec_unit = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA_DIR, 'fichas_tecnicas_flat.csv'))):
        try:
            q, pax, cp = float(r['qtd']), float(r['pax']), float(r['custo_pax'])
        except (ValueError, TypeError):
            continue
        if q > 0 and pax > 0 and cp > 0:
            rec_unit[r['erp_ref']].append(cp * pax / q)
    return lines, dayvol, {k: float(np.median(v)) for k, v in rec_unit.items()}


FEATS = ['qty', 'log_qty', 'value', 'unit_cost', 'z_hist', 'abs_z_hist',
         'qty_share_day', 'art_freq', 'art_seen', 'cost_div_recipe',
         'dow', 'week', 'fam', 'wh', 'stream']


def build(train, test, dayvol, rec_unit):
    """Ajusta estatisticas SO no treino e devolve matrizes ja escaladas."""
    stats, freq, byref = {}, Counter(), defaultdict(list)
    for l in train:
        byref[l['ref']].append(l['qty']); freq[l['ref']] += 1
    for k, v in byref.items():
        stats[k] = (float(np.mean(v)), float(np.std(v)) if len(v) > 1 else 0.0)
    cats = {}

    def ci(name, val):
        d = cats.setdefault(name, {})
        return d.setdefault(val, len(d))

    def row(l):
        m, s = stats.get(l['ref'], (np.nan, np.nan))
        z = (l['qty'] - m) / s if (s and s > 0) else np.nan
        vq = dayvol.get((l['stream'], l['date']), (np.nan, np.nan))[0]
        ru = rec_unit.get(l['ref'])
        cd = (l['unit_cost'] - ru) / ru if (ru and ru > 0 and l['unit_cost'] > 0) else np.nan
        return [l['qty'], math.log1p(max(l['qty'], 0)), l['qty'] * l['unit_cost'],
                l['unit_cost'], z, abs(z) if z == z else np.nan,
                l['qty'] / vq if (vq and vq > 0) else np.nan,
                freq.get(l['ref'], 0), 1.0 if l['ref'] in stats else 0.0, cd,
                l['date'].weekday(), (l['date'] - WINDOW_START).days // 7,
                ci('fam', l['ref'][:2]), ci('wh', l['wh']), ci('stream', l['stream'])]

    A = np.array([row(l) for l in train], dtype=float)
    B = np.array([row(l) for l in test], dtype=float)
    med = np.nanmedian(A, axis=0)
    keep = np.isnan(A).sum(axis=0) > 0

    def prep(X):
        miss = np.isnan(X).astype(float)
        return np.hstack([np.where(np.isnan(X), med, X), miss[:, keep]])

    sc = StandardScaler().fit(prep(A))
    names = FEATS + ['%s_missing' % FEATS[i] for i in np.where(keep)[0]]
    return sc.transform(prep(A)), sc.transform(prep(B)), names, A
