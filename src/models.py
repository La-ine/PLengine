
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

from src.features import FEATS, FeatureBuilder

LAB = {"A": 0, "D": 1, "H": 2}   # class order used everywhere: Away, Draw, Home
INV = {v: k for k, v in LAB.items()}

# ---------- metrics ----------
def rps(p, y):
    cp = np.cumsum(p, 1)
    cy = np.cumsum(np.eye(3)[y], 1)
    return float((((cp - cy) ** 2)[:, :2]).sum(1).mean() / 2)

def metrics(y, p):
    p = normalize_proba(p)

    return {
        "acc": accuracy_score(y, p.argmax(axis=1)),
        "logloss": log_loss(y, p, labels=[0, 1, 2]),
        "rps": rps(p, y),
    }

# ---------- Dixon-Coles ----------
class DixonColes:
    def __init__(self, half_life_days=730, max_goals=8, reg=0.005):
        self.half_life, self.G, self.reg = half_life_days, max_goals, reg

    def fit(self, df):
        teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        self.idx = {t: i for i, t in enumerate(teams)}
        N = len(teams)
        hi = df.HomeTeam.map(self.idx).values
        ai = df.AwayTeam.map(self.idx).values
        hg, ag = df.FTHG.values.astype(float), df.FTAG.values.astype(float)
        age = (df.Date.max() - df.Date).dt.days.values
        w = 0.5 ** (age / self.half_life)
        m00, m01, m10, m11 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

        def nll(th):
            att = th[:N] - th[:N].mean()
            dfn = th[N:2 * N]
            home, rho = th[2 * N], th[2 * N + 1]
            lam = np.exp(att[hi] + dfn[ai] + home)
            mu = np.exp(att[ai] + dfn[hi])
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-6, None)
            ll = np.log(tau) + hg * np.log(lam) - lam + ag * np.log(mu) - mu
            return -(w * ll).sum() + self.reg * (th[:2 * N] ** 2).sum()

        x0 = np.zeros(2 * N + 2); x0[2 * N] = 0.25
        bounds = [(None, None)] * (2 * N) + [(None, None), (-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        th = res.x
        att = th[:N] - th[:N].mean()
        self.att = dict(zip(teams, att))
        self.dfn = dict(zip(teams, th[N:2 * N]))
        self.home, self.rho = th[2 * N], th[2 * N + 1]
        return self

    def rates(self, h, a):
        lam = np.exp(self.att.get(h, 0.0) + self.dfn.get(a, 0.0) + self.home)
        mu = np.exp(self.att.get(a, 0.0) + self.dfn.get(h, 0.0))
        return lam, mu

    def score_matrix(self, h, a):
        lam, mu = self.rates(h, a)
        g = np.arange(self.G + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        M[0, 0] *= max(1 - lam * mu * self.rho, 1e-6)
        M[0, 1] *= max(1 + lam * self.rho, 1e-6)
        M[1, 0] *= max(1 + mu * self.rho, 1e-6)
        M[1, 1] *= max(1 - self.rho, 1e-6)
        return M / M.sum()

    def proba(self, h, a):
        M = self.score_matrix(h, a)
        return np.array([np.triu(M, 1).sum(), np.trace(M), np.tril(M, -1).sum()])  # A, D, H

    def predict_proba(self, df):
        return np.vstack([self.proba(h, a) for h, a in zip(df.HomeTeam, df.AwayTeam)])

# ---------- XGBoost ----------
DEFAULT_XGB = dict(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0)

def fit_xgb(X, y, params=None):
    p = {**DEFAULT_XGB, **(params or {})}
    m = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                          tree_method="hist", n_jobs=-1, random_state=42, **p)
    return m.fit(X, y)

# ---------- walk-forward helpers ----------
def dc_walk_forward(df, season, half_life, refit_days=None):
    """Predict every match of `season`, refitting DC on data strictly before each chunk."""
    test = df[df.season == season]
    start, end = test.Date.min(), test.Date.max()
    out = pd.DataFrame(index=test.index, columns=range(3), dtype=float)
    cut = start
    step = pd.Timedelta(days=refit_days) if refit_days else (end - start + pd.Timedelta(days=2))
    while cut <= end:
        chunk = test[(test.Date >= cut) & (test.Date < cut + step)]
        if len(chunk):
            m = DixonColes(half_life).fit(df[df.Date < cut])
            out.loc[chunk.index] = m.predict_proba(chunk)
        cut += step
    return out.values

def xgb_walk_forward(feat, season, min_train_season, params=None):
    test = feat[feat.season == season]
    train = feat[(feat.Date < test.Date.min()) & (feat.season >= min_train_season)]
    m = fit_xgb(train[FEATS], train.FTR.map(LAB), params)
    return m.predict_proba(test[FEATS])

def normalize_proba(p):
    p = np.asarray(p, dtype=np.float64)

    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected shape (n_matches, 3), got {p.shape}")

    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN or infinity")

    if (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must lie between 0 and 1")

    totals = p.sum(axis=1, keepdims=True)

    if (totals <= 0).any():
        raise ValueError("A probability row has zero total")

    max_error = float(np.max(np.abs(totals - 1.0)))
    if max_error > 1e-5:
        raise ValueError(
            f"Probabilities are not normalized: maximum error = {max_error:.3g}"
        )

    return p / totals

def blend(p_dc, p_xgb, w):
    if not 0 <= w <= 1:
        raise ValueError("Blend weight must be between 0 and 1")

    p_dc = normalize_proba(p_dc)
    p_xgb = normalize_proba(p_xgb)

    return normalize_proba(w * p_dc + (1 - w) * p_xgb)

def fit_final(df, feat, min_train_season, params):
    """Fit both models on everything available. Returns (dc, xgb_model, w)."""
    dc = DixonColes(params.get("dc_half_life", 730)).fit(df)
    xp = {k: v for k, v in params.items() if k not in ("w_dc", "dc_half_life")}
    train = feat[feat.season >= min_train_season]
    mx = fit_xgb(train[FEATS], train.FTR.map(LAB), xp)
    return dc, mx, params.get("w_dc", 0.5)
