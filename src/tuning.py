
import numpy as np
import optuna
from sklearn.metrics import log_loss

from src.models import (LAB, DixonColes, dc_walk_forward, xgb_walk_forward, fit_xgb, blend)

optuna.logging.set_verbosity(optuna.logging.WARNING)
HALF_LIVES = [180, 365, 730, 1500]

def tune_dc(df, val_seasons):
    """Tiny grid search: Dixon-Coles has only one hyperparameter that matters (time decay)."""
    y = np.concatenate([df[df.season == s].FTR.map(LAB).values for s in val_seasons])
    scores = {}
    for hl in HALF_LIVES:
        p = np.vstack([dc_walk_forward(df, s, hl) for s in val_seasons])
        scores[hl] = log_loss(y, p, labels=[0, 1, 2])
    best = min(scores, key=scores.get)
    return best, scores

def tune_xgb_and_blend(df, feat, val_seasons, min_train_season, dc_half_life, n_trials=30):
    """Optuna searches XGBoost hyperparameters AND the Dixon-Coles/XGBoost blend weight."""
    y = np.concatenate([feat[feat.season == s].FTR.map(LAB).values for s in val_seasons])
    p_dc = np.vstack([dc_walk_forward(df, s, dc_half_life) for s in val_seasons])

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        )
        w = trial.suggest_float("w_dc", 0.0, 1.0)
        p_x = np.vstack([xgb_walk_forward(feat, s, min_train_season, params) for s in val_seasons])
        return log_loss(y, blend(p_dc, p_x, w), labels=[0, 1, 2])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value
