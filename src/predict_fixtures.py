
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import download, load
from src.features import FeatureBuilder, FEATS
from src.fixtures import get_fixtures
from src.models import fit_final, blend, INV, DEFAULT_XGB

START = 2010
OUTDIR = ROOT / "outputs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="only predict the next N days")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)

    if not a.no_download:
        download(start=START)
    df = load(start=START)

    pf = OUTDIR / "best_params.json"
    params = json.loads(pf.read_text()) if pf.exists() else {"w_dc": 0.5, "dc_half_life": 730,
                                                              **DEFAULT_XGB}
    if not pf.exists():
        print("No best_params.json found: using defaults (run src/run.py first).")
    min_train = params.pop("min_train_season", sorted(df.season.unique())[1])

    fb = FeatureBuilder()
    feat = fb.transform(df)                      # fb now holds the CURRENT Elo/form state
    dc, mx, w = fit_final(df, feat, min_train, params)

    fx = get_fixtures(set(df.HomeTeam) | set(df.AwayTeam), ROOT / "data" / "fixtures_manual.csv",
                      limit_days=a.days)
    if fx.empty:
        raise SystemExit("No upcoming fixtures in the selected window.")
    
    fixtures_path = ROOT / "data" / "fixtures_upcoming.csv"
    fx.to_csv(fixtures_path, index=False)
    print(f"Saved downloaded fixtures to {fixtures_path}")

    X = pd.DataFrame([fb.row(h, t) for h, t in zip(fx.HomeTeam, fx.AwayTeam)])[FEATS]
    p_dc = np.vstack([dc.proba(h, t) for h, t in zip(fx.HomeTeam, fx.AwayTeam)])
    p = blend(p_dc, mx.predict_proba(X), w)

    xg = np.array([dc.rates(h, t) for h, t in zip(fx.HomeTeam, fx.AwayTeam)])
    top = []
    for h, t in zip(fx.HomeTeam, fx.AwayTeam):
        M = dc.score_matrix(h, t)
        i, j = np.unravel_index(M.argmax(), M.shape)
        top.append(f"{i}-{j}")

    out = fx.assign(p_home=p[:, 2], p_draw=p[:, 1], p_away=p[:, 0],
                    pick=[{"H": "Home", "D": "Draw", "A": "Away"}[INV[k]] for k in p.argmax(1)],
                    exp_goals_home=xg[:, 0].round(2), exp_goals_away=xg[:, 1].round(2),
                    likely_score=top)
    out.to_csv(OUTDIR / "predictions.csv", index=False)
    print("\n" + out.round(3).to_string(index=False))
    print(f"\nSaved {len(out)} predictions to outputs/predictions.csv")

if __name__ == "__main__":
    main()
