import numpy as np, xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score

FEATS = ["elo_diff","elo_h","elo_a","form_h","form_a","gd_h","gd_a"]
LAB = {"A":0,"D":1,"H":2}

def rps(p, y):
    cp, cy = np.cumsum(p,1), np.cumsum(np.eye(3)[y],1)
    return ((cp-cy)**2)[:, :2].sum(1).mean()/2

def walk_forward(df, test_seasons):
    out = []
    for s in test_seasons:
        tr, te = df[df.season < s], df[df.season == s]
        m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="multi:softprob")
        m.fit(tr[FEATS], tr.FTR.map(LAB))
        p, y = m.predict_proba(te[FEATS]), te.FTR.map(LAB).values
        out.append((s, accuracy_score(y,p.argmax(1)), log_loss(y,p), rps(p,y)))
    return out