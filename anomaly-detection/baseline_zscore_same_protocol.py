# -*- coding: utf-8 -*-
"""Baseline |z| > k versus Isolation Forest under EXACTLY the protocol that produced
the project's primary result (check_augmented_temporal.py, variant C: temporal hold-out,
train <= 14/06, test 15-28/06, z-score baseline recomputed on TRAIN only, same rows,
IsolationForest n_estimators=200, contamination = train anomaly rate, seed 42).
Supervisor review 23/09/2026, point 3. Replaces baseline_zscore_temporal.py, whose row
filtering and best-F1 threshold made the IF row differ slightly from the project report.
"""
import numpy as np, pandas as pd, datetime, os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
SEED=42; TRAIN_END=datetime.date(2026,6,14)
P=os.environ.get('AUGMENTED_CSV', os.path.join(os.path.dirname(os.path.abspath(__file__)),'augmented_real_dataset.csv'))
df=pd.read_csv(P)
df['deviation_zscore_historical']=df['deviation_zscore_historical'].replace([np.inf,-np.inf],np.nan)
df=df.dropna(subset=['deviation_zscore_historical']).reset_index(drop=True)
df['dt']=pd.to_datetime(df['Data'],format='%d.%m.%Y',errors='coerce').dt.date
d2=df.dropna(subset=['dt']).copy()
g=d2[d2['dt']<=TRAIN_END].groupby('Ref')['Quant_original']; m,s=g.mean(),g.std()
d2['z']=((d2['Quant']-d2['Ref'].map(m))/d2['Ref'].map(s)).replace([np.inf,-np.inf],np.nan)
d2=d2.dropna(subset=['z'])
tr=d2[d2['dt']<=TRAIN_END]; te=d2[d2['dt']>TRAIN_END]
ytr,yte=tr['label'].values,te['label'].values; rate=ytr.mean()
print('train %d (%d anom) | test %d (%d anom, base rate %.4f)'%(len(tr),ytr.sum(),len(te),yte.sum(),yte.mean()))
def rep(name,thr_rule,pred,score):
    p,r,f=precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0)
    ap=average_precision_score(yte,score)
    print('%-34s %-26s P %.3f  R %.3f  F1 %.3f  PR-AUC %.3f  flagged %d'%(name,thr_rule,p,r,f,ap,pred.sum()))
    return dict(model=name,threshold=thr_rule,precision=p,recall=r,f1=f,prauc=ap,flagged=int(pred.sum()))
def bestf1(y,sc):
    p,r,t=precision_recall_curve(y,sc); f=2*p*r/np.maximum(p+r,1e-12); return t[np.argmax(f[:-1])]
sc=StandardScaler().fit(tr[['z']]); A,B=sc.transform(tr[['z']]),sc.transform(te[['z']])
iso=IsolationForest(n_estimators=200,contamination=rate,random_state=SEED).fit(A)
s_tr,s_te=-iso.decision_function(A),-iso.decision_function(B)
az_tr,az_te=np.abs(tr['z'].values),np.abs(te['z'].values)
res=[]
res.append(rep('Isolation Forest','contamination (default)',(iso.predict(B)==-1).astype(int),s_te))
res.append(rep('Isolation Forest','best-F1 on train',(s_te>=bestf1(ytr,s_tr)).astype(int),s_te))
k_c=np.quantile(az_tr,1-rate)
res.append(rep('|z| rule','contamination k=%.2f'%k_c,(az_te>=k_c).astype(int),az_te))
k_f=bestf1(ytr,az_tr)
res.append(rep('|z| rule','best-F1 on train k=%.2f'%k_f,(az_te>=k_f).astype(int),az_te))
for k in (2,3): res.append(rep('|z| rule','fixed k=%d'%k,(az_te>=k).astype(int),az_te))
pd.DataFrame(res).to_csv('baseline_zscore_same_protocol_results.csv',index=False)
