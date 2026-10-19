import numpy as np, pandas as pd
from collections import defaultdict, deque

def build_features(df, k=20, home_adv=60, window=5):
    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: deque(maxlen=window))   # points
    gd   = defaultdict(lambda: deque(maxlen=window))   # goal diff
    rows = []
    for r in df.itertuples():
        h, a = r.HomeTeam, r.AwayTeam
        eh, ea = elo[h], elo[a]
        rows.append({
            "elo_h": eh, "elo_a": ea, "elo_diff": eh - ea + home_adv,
            "form_h": np.mean(form[h]) if form[h] else 1.4,
            "form_a": np.mean(form[a]) if form[a] else 1.4,
            "gd_h": np.mean(gd[h]) if gd[h] else 0,
            "gd_a": np.mean(gd[a]) if gd[a] else 0,
        })
        res = {"H":1,"D":.5,"A":0}[r.FTR]
        exp = 1/(1+10**(-(eh-ea+home_adv)/400))
        mov = np.log(abs(r.FTHG-r.FTAG)+1)+1          # margin multiplier
        elo[h] += k*mov*(res-exp); elo[a] -= k*mov*(res-exp)
        ph, pa = {"H":(3,0),"D":(1,1),"A":(0,3)}[r.FTR]
        form[h].append(ph); form[a].append(pa)
        gd[h].append(r.FTHG-r.FTAG); gd[a].append(r.FTAG-r.FTHG)
    return pd.concat([df, pd.DataFrame(rows)], axis=1)