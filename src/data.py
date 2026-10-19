import pandas as pd

def season_codes(start=2014, end=2026):
    return [f"{y%100:02d}{(y+1)%100:02d}" for y in range(start, end)]

def load(start=2014, end=2026):
    dfs = []
    for s in season_codes(start, end):
        url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
        try:
            d = pd.read_csv(url, encoding="latin1")
            d["season"] = s
            dfs.append(d)
        except Exception as e:
            print("skip", s, e)
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    cols = ["season","Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
            "HS","AS","HST","AST"]
    return df[cols].dropna(subset=["Date","FTR"]).sort_values("Date").reset_index(drop=True)