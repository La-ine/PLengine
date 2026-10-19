from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "matches.csv"
CURRENT = "2627"
COLS = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","HS","AS","HST","AST"]

def season_codes(start=2014, end=2027):
    return [f"{y%100:02d}{(y+1)%100:02d}" for y in range(start, end)]

def download(start=2014, end=2027, force_current=True):
    RAW.mkdir(parents=True, exist_ok=True)
    for s in season_codes(start, end):
        path = RAW / f"E0_{s}.csv"
        if path.exists() and not (force_current and s == CURRENT):
            continue
        url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
        try:
            pd.read_csv(url, encoding="latin1").to_csv(path, index=False)
            print("saved", path)
        except Exception as e:
            print("skip", s, e)

def load(start=2014, end=2027):
    dfs = []
    for s in season_codes(start, end):
        path = RAW / f"E0_{s}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[[c for c in COLS if c in d.columns]].assign(season=s)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date","FTR"]).sort_values("Date").reset_index(drop=True)

if __name__ == "__main__":
    download()
    df = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.shape, df.season.nunique(), "seasons")
    print(df.tail())