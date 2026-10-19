
import difflib, io, os
import pandas as pd
import requests

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves", "Brighton & Hove Albion": "Brighton",
    "Leeds United": "Leeds", "Leicester City": "Leicester", "AFC Bournemouth": "Bournemouth",
    "Ipswich Town": "Ipswich", "Luton Town": "Luton", "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom", "Sheffield United": "Sheffield United",
    "Sunderland AFC": "Sunderland", "Coventry City": "Coventry", "Hull City": "Hull",
}

def map_team(name, known):
    n = name.replace(" FC", "").replace(" AFC", "").strip()
    n = ALIASES.get(n, ALIASES.get(name, n))
    if n in known:
        return n
    close = difflib.get_close_matches(n, list(known), n=1, cutoff=0.6)
    if close:
        print(f"  name match: '{name}' -> '{close[0]}'")
        return close[0]
    print(f"  WARNING: unknown team '{name}' (will be rated as an average side)")
    return n

def from_csv():
    r = requests.get(FIXTURES_URL, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")), on_bad_lines="skip")
    d = d[d["Div"] == "E0"]
    d["Date"] = pd.to_datetime(d["Date"], format="mixed", dayfirst=True, errors="coerce")
    return d[["Date", "HomeTeam", "AwayTeam"]].dropna()

def from_api():
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("no FOOTBALL_DATA_TOKEN set")
    r = requests.get(API_URL, params={"status": "SCHEDULED,TIMED"},
                     headers={"X-Auth-Token": token}, timeout=30)
    r.raise_for_status()
    m = r.json()["matches"]
    return pd.DataFrame({
        "Date": pd.to_datetime([x["utcDate"] for x in m]).tz_localize(None),
        "HomeTeam": [x["homeTeam"]["name"] for x in m],
        "AwayTeam": [x["awayTeam"]["name"] for x in m],
    })

def from_manual(path):
    d = pd.read_csv(path)
    d["Date"] = pd.to_datetime(d["Date"], format="mixed", dayfirst=True)
    return d[["Date", "HomeTeam", "AwayTeam"]]

def get_fixtures(known_teams, manual_path, limit_days=None):
    """Try football-data.co.uk, then football-data.org API, then a hand-written CSV."""
    d = None
    sources = [
        ("football-data.org API", from_api),
        ("football-data.co.uk fixtures.csv", from_csv),
        ("manual file", lambda: from_manual(manual_path)),
    ]

    for name, fn in sources:
        try:
            d = fn()
            if d is not None and len(d):
                print(f"fixtures source: {name} ({len(d)} matches)")
                break
            print(f"  {name}: no Premier League rows")
        except Exception as e:
            print(f"  {name}: failed ({e})")
            d = None
    if d is None or not len(d):
        raise SystemExit("No fixtures found. Put them in data/fixtures_manual.csv "
                         "(columns: Date,HomeTeam,AwayTeam) or set FOOTBALL_DATA_TOKEN.")
    d = d[d.Date >= pd.Timestamp.now().normalize()].sort_values("Date")
    if limit_days:
        d = d[d.Date <= pd.Timestamp.now().normalize() + pd.Timedelta(days=limit_days)]
    d = d.assign(HomeTeam=[map_team(t, known_teams) for t in d.HomeTeam],
                 AwayTeam=[map_team(t, known_teams) for t in d.AwayTeam])
    return d.reset_index(drop=True)
