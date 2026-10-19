
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import download, load
from src.features import FeatureBuilder
from src.models import (LAB, metrics, DixonColes, dc_walk_forward, xgb_walk_forward, blend)
from src.tuning import tune_dc, tune_xgb_and_blend

START = 2010
OUTDIR = ROOT / "outputs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)

    if not a.no_download:
        download(start=START)
    df = load(start=START)
    feat = FeatureBuilder().transform(df)

    sizes = df.groupby("season").size()
    complete = [s for s in sorted(sizes.index) if sizes[s] >= 380]
    test_seasons, val_seasons = complete[-2:], complete[-4:-2]
    min_train = sorted(sizes.index)[1]          # first season = Elo warm-up, not used for training
    print(f"{len(df)} matches | warm-up: {sorted(sizes.index)[0]} | val: {val_seasons} | test: {test_seasons}")

    print("\n[1/3] Tuning Dixon-Coles time decay (grid)...")
    hl, hl_scores = tune_dc(df, val_seasons)
    print("  val log-loss by half-life (days):", {k: round(v, 4) for k, v in hl_scores.items()}, "-> best", hl)

    print(f"\n[2/3] Optuna: {a.trials} trials (XGBoost params + blend weight)...")
    best, val_ll = tune_xgb_and_blend(df, feat, val_seasons, min_train, hl, a.trials)
    w = best.pop("w_dc")
    print(f"  best val log-loss {val_ll:.4f} | blend weight on Dixon-Coles = {w:.2f}")
    print("  best XGB params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})

    print("\n[3/3] Final backtest on unseen seasons (walk-forward, params frozen)...")
    y = np.concatenate([feat[feat.season == s].FTR.map(LAB).values for s in test_seasons])
    p_dc = np.vstack([dc_walk_forward(df, s, hl) for s in test_seasons])
    p_xd = np.vstack([xgb_walk_forward(feat, s, min_train) for s in test_seasons])
    p_xt = np.vstack([xgb_walk_forward(feat, s, min_train, best) for s in test_seasons])
    p_bl = blend(p_dc, p_xt, w)

    prior_predictions = []

    for season in test_seasons:
        test = feat[feat.season == season]

        train = feat[(feat.Date < test.Date.min()) & (feat.season >= min_train)]

        prior = (
            train.FTR.map(LAB)
            .value_counts(normalize=True)
            .reindex([0, 1, 2], fill_value=0.0)
            .to_numpy()
        )

        prior_predictions.append(np.tile(prior, (len(test), 1)))

    p_prior = np.vstack(prior_predictions)
    p_home = np.tile([0.05, 0.05, 0.90], (len(y), 1))

    rows = {"Always home win": p_home, "Class frequencies": p_prior, "Dixon-Coles": p_dc,
            "XGBoost (default)": p_xd, "XGBoost (Optuna)": p_xt, "Blend DC+XGB": p_bl}
    results = {}
    for name, probabilities in rows.items():
        try:
            results[name] = metrics(y, probabilities)
        except ValueError as exc:
            raise ValueError(f"Evaluation failed for {name!r}: {exc}") from exc

    res = pd.DataFrame.from_dict(results, orient="index")
    
    res.columns = ["Accuracy", "Log-loss", "RPS"]
    print("\n" + res.round(4).to_string())
    res.to_csv(OUTDIR / "backtest_results.csv")

    params = {**best, "w_dc": w, "dc_half_life": hl, "min_train_season": min_train}
    (OUTDIR / "best_params.json").write_text(json.dumps(params, indent=2))
    print("\nSaved outputs/backtest_results.csv and outputs/best_params.json")

if __name__ == "__main__":
    main()
