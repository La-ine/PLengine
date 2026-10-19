
from collections import defaultdict, deque
import numpy as np
import pandas as pd

AVG = 1.4

class FeatureBuilder:
    """Elo + rolling form. Features for a match are computed BEFORE its result is used."""

    def __init__(self, k=20, home_adv=60, window=5, carry=0.75):
        self.k, self.home_adv, self.window, self.carry = k, home_adv, window, carry
        self.elo = defaultdict(lambda: 1500.0)
        mk = lambda: defaultdict(lambda: deque(maxlen=window))
        self.pts, self.gd, self.gf, self.ga = mk(), mk(), mk(), mk()

    @staticmethod
    def _m(d, default):
        return float(np.mean(d)) if len(d) else default

    def new_season(self):
        for t in list(self.elo):
            self.elo[t] = self.carry * self.elo[t] + (1 - self.carry) * 1500.0

    def row(self, h, a):
        eh, ea = self.elo[h], self.elo[a]
        return {
            "elo_h": eh, "elo_a": ea, "elo_diff": eh - ea + self.home_adv,
            "form_h": self._m(self.pts[h], AVG), "form_a": self._m(self.pts[a], AVG),
            "gd_h": self._m(self.gd[h], 0.0), "gd_a": self._m(self.gd[a], 0.0),
            "gf_h": self._m(self.gf[h], AVG), "gf_a": self._m(self.gf[a], AVG),
            "ga_h": self._m(self.ga[h], AVG), "ga_a": self._m(self.ga[a], AVG),
        }

    def update(self, h, a, hg, ag, ftr):
        eh, ea = self.elo[h], self.elo[a]
        res = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
        exp = 1 / (1 + 10 ** (-(eh - ea + self.home_adv) / 400))
        mov = np.log(abs(hg - ag) + 1) + 1
        delta = self.k * mov * (res - exp)
        self.elo[h] += delta
        self.elo[a] -= delta
        ph, pa = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[ftr]
        self.pts[h].append(ph); self.pts[a].append(pa)
        self.gd[h].append(hg - ag); self.gd[a].append(ag - hg)
        self.gf[h].append(hg); self.gf[a].append(ag)
        self.ga[h].append(ag); self.ga[a].append(hg)

    def transform(self, df):
        rows, prev = [], None
        for r in df.itertuples():
            if prev is not None and r.season != prev:
                self.new_season()
            prev = r.season
            rows.append(self.row(r.HomeTeam, r.AwayTeam))
            self.update(r.HomeTeam, r.AwayTeam, r.FTHG, r.FTAG, r.FTR)
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

FEATS = ["elo_diff", "elo_h", "elo_a", "form_h", "form_a", "gd_h", "gd_a",
         "gf_h", "gf_a", "ga_h", "ga_a"]
