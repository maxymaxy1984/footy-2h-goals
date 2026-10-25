"""2nd-Half Goal Finder — single-file version.  Run:  python scan.py"""
from __future__ import annotations
import sys
C = F = sys.modules[__name__]   # sections below share one namespace

# ======================================================================
# config
# ======================================================================
"""Central configuration. Edit here or override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DOCS = ROOT / "docs"          # GitHub Pages output

# ---------------------------------------------------------------------------
# Domestic leagues with half-time scores (football-data.co.uk codes)
# ---------------------------------------------------------------------------
LEAGUES = {
    "E0": "England Premier League", "E1": "England Championship",
    "E2": "England League One", "E3": "England League Two",
    "EC": "England National League",
    "SC0": "Scotland Premiership", "SC1": "Scotland Championship",
    "SC2": "Scotland League One", "SC3": "Scotland League Two",
    "D1": "Germany Bundesliga", "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A", "I2": "Italy Serie B",
    "SP1": "Spain La Liga", "SP2": "Spain Segunda",
    "F1": "France Ligue 1", "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie", "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga", "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
}

# How many past seasons of history to use (current season included).
SEASONS_BACK = int(os.getenv("H2_SEASONS_BACK", "8"))

# ---------------------------------------------------------------------------
# Cup / European competitions (optional, need free API keys)
# ---------------------------------------------------------------------------
# football-data.org free tier: Champions League (+ domestic top flights).
FOOTBALL_DATA_ORG_KEY = os.getenv("FOOTBALL_DATA_ORG_KEY", "")
FDORG_CUPS = {"CL": "UEFA Champions League"}

# API-Football (api-sports.io): Europa League, Conference League, domestic cups.
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
APIF_CUPS = {
    3: "UEFA Europa League", 848: "UEFA Conference League",
    45: "England FA Cup", 48: "England League Cup",
    81: "Germany DFB Pokal", 137: "Italy Coppa Italia",
    143: "Spain Copa del Rey", 66: "France Coupe de France",
    90: "Netherlands KNVB Beker", 96: "Portugal Taca de Portugal",
    147: "Belgium Cup", 206: "Turkey Cup", 181: "Scotland FA Cup",
}

# ---------------------------------------------------------------------------
# Model + selection settings
# ---------------------------------------------------------------------------
TEAM_HALFLIFE = 19        # matches; how fast team form decays
LEAGUE_HALFLIFE = 380     # matches; league scoring baseline drift
SHRINK_K = 6.0            # pseudo-matches of league-average prior per team
MIN_TEAM_MATCHES = 8      # both teams need this many rated matches
PICK_THRESHOLD = float(os.getenv("H2_THRESHOLD", "0.86"))  # calibrated P(2H goal)
DAYS_AHEAD = int(os.getenv("H2_DAYS_AHEAD", "3"))

# Optional Telegram alert
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ======================================================================
# fetch
# ======================================================================
"""Data acquisition.

All sources are normalised to one match schema:
    date, comp, comp_name, kind ('league'|'cup'), home, away,
    hthg, htag, fthg, ftag          (90-minute scores; NaN for fixtures)
Past seasons are cached on disk and never re-downloaded; the current season
is refreshed on every run.
"""

import io
import json
import logging
import os
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests


log = logging.getLogger(__name__)
FD_BASE = "https://www.football-data.co.uk"
UA = {"User-Agent": "h2goals/1.0 (personal research)"}
COLS = ["date", "comp", "comp_name", "kind", "home", "away",
        "hthg", "htag", "fthg", "ftag"]


def season_start(d: date | None = None) -> int:
    d = d or today()
    return d.year if d.month >= 7 else d.year - 1


def fd_code(start: int) -> str:
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def today() -> date:
    o = os.getenv("H2_TODAY")          # for testing / replays
    return date.fromisoformat(o[:10]) if o else date.today()


def _get(url, headers=None, params=None, tries=3):
    if os.getenv("H2_OFFLINE"):
        return None
    for i in range(tries):
        try:
            r = requests.get(url, headers={**UA, **(headers or {})},
                             params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(15 * (i + 1)); continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning("GET %s failed (%s), attempt %d", url, e, i + 1)
            time.sleep(3 * (i + 1))
    return None


def _cached(path, fetch_fn, refresh: bool):
    if path.exists() and not refresh:
        return path.read_bytes()
    content = fetch_fn()
    if content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return content
    return path.read_bytes() if path.exists() else None


# ---------------------------------------------------------------------------
# football-data.co.uk  (domestic leagues, HT+FT, back to the 1990s)
# ---------------------------------------------------------------------------
def _parse_fdcouk(raw: bytes, code: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    need = {"Date", "HomeTeam", "AwayTeam"}
    if not need.issubset(df.columns):
        # a few old files use HT/AT
        df = df.rename(columns={"HT": "HomeTeam", "AT": "AwayTeam"})
        if not need.issubset(df.columns):
            return pd.DataFrame(columns=COLS)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "comp": df.get("Div", code),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
    })
    for src, dst in [("HTHG", "hthg"), ("HTAG", "htag"),
                     ("FTHG", "fthg"), ("FTAG", "ftag")]:
        out[dst] = pd.to_numeric(df.get(src), errors="coerce")
    out["comp"] = code
    out["comp_name"] = C.LEAGUES.get(code, code)
    out["kind"] = "league"
    return out.dropna(subset=["date"])[COLS]


def fetch_league_history(seasons_back=C.SEASONS_BACK) -> pd.DataFrame:
    cur = season_start()
    frames = []
    for start in range(cur - seasons_back + 1, cur + 1):
        sc = fd_code(start)
        for code in C.LEAGUES:
            path = C.RAW / "fdcouk" / sc / f"{code}.csv"
            url = f"{FD_BASE}/mmz4281/{sc}/{code}.csv"
            raw = _cached(path, lambda: (r := _get(url)) and r.content,
                          refresh=(start == cur))
            if raw:
                frames.append(_parse_fdcouk(raw, code))
    hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    return hist.dropna(subset=["hthg", "htag", "fthg", "ftag"])


def fetch_league_fixtures() -> pd.DataFrame:
    path = C.RAW / "fdcouk" / "fixtures.csv"
    raw = _cached(path, lambda: (r := _get(f"{FD_BASE}/fixtures.csv")) and r.content,
                  refresh=True)
    if not raw:
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.rename(columns=lambda c: str(c).replace("﻿", "").replace("ï»¿", ""))
    df = df[df["Div"].isin(C.LEAGUES)]
    when = df["Date"].astype(str) + " " + df.get("Time", "15:00").astype(str)
    out = pd.DataFrame({
        "date": pd.to_datetime(when, dayfirst=True, errors="coerce"),
        "comp": df["Div"], "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
    })
    out["comp_name"] = out["comp"].map(C.LEAGUES)
    out["kind"] = "league"
    for c in ["hthg", "htag", "fthg", "ftag"]:
        out[c] = float("nan")
    return out.dropna(subset=["date"])[COLS]


# ---------------------------------------------------------------------------
# football-data.org  (Champions League; free key)
# ---------------------------------------------------------------------------
def _fdorg(comp: str, start: int, refresh: bool):
    path = C.RAW / "fdorg" / f"{comp}_{start}.json"
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches"

    def go():
        r = _get(url, headers={"X-Auth-Token": C.FOOTBALL_DATA_ORG_KEY},
                 params={"season": start})
        time.sleep(6.5)  # free tier: 10 requests/minute
        return r and r.content
    raw = _cached(path, go, refresh)
    return json.loads(raw) if raw else {"matches": []}


def fetch_fdorg_cups(seasons_back=4):
    if not C.FOOTBALL_DATA_ORG_KEY:
        return pd.DataFrame(columns=COLS)
    cur, rows = season_start(), []
    for comp, name in C.FDORG_CUPS.items():
        for start in range(cur - seasons_back + 1, cur + 1):
            for m in _fdorg(comp, start, refresh=(start == cur)).get("matches", []):
                s = m.get("score", {})
                ft = s.get("regularTime") if s.get("duration") != "REGULAR" and s.get("regularTime") else s.get("fullTime")
                ht = s.get("halfTime") or {}
                done = m.get("status") == "FINISHED"
                rows.append({
                    "date": pd.to_datetime(m["utcDate"]).tz_convert(None),
                    "comp": comp, "comp_name": name, "kind": "cup",
                    "home": (m["homeTeam"] or {}).get("name"),
                    "away": (m["awayTeam"] or {}).get("name"),
                    "hthg": ht.get("home") if done else None,
                    "htag": ht.get("away") if done else None,
                    "fthg": (ft or {}).get("home") if done else None,
                    "ftag": (ft or {}).get("away") if done else None,
                })
    df = pd.DataFrame(rows, columns=COLS)
    return df.dropna(subset=["home", "away"])


# ---------------------------------------------------------------------------
# API-Football  (Europa/Conference League + domestic cups; free key)
# ---------------------------------------------------------------------------
def fetch_apif_cups(seasons_back=3):
    if not C.API_FOOTBALL_KEY:
        return pd.DataFrame(columns=COLS)
    cur, rows = season_start(), []
    for lid, name in C.APIF_CUPS.items():
        for start in range(cur - seasons_back + 1, cur + 1):
            path = C.RAW / "apif" / f"{lid}_{start}.json"

            def go(lid=lid, start=start):
                r = _get("https://v3.football.api-sports.io/fixtures",
                         headers={"x-apisports-key": C.API_FOOTBALL_KEY},
                         params={"league": lid, "season": start})
                return r and r.content
            raw = _cached(path, go, refresh=(start == cur))
            if not raw:
                continue
            payload = json.loads(raw)
            if payload.get("errors"):
                log.warning("API-Football %s %s: %s", name, start, payload["errors"])
            for f in payload.get("response", []):
                st = f["fixture"]["status"]["short"]
                done = st in ("FT", "AET", "PEN")
                sc = f.get("score", {})
                rows.append({
                    "date": pd.to_datetime(f["fixture"]["date"]).tz_convert(None),
                    "comp": f"AF{lid}", "comp_name": name, "kind": "cup",
                    "home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"],
                    "hthg": sc["halftime"]["home"] if done else None,
                    "htag": sc["halftime"]["away"] if done else None,
                    "fthg": sc["fulltime"]["home"] if done else None,   # 90 min
                    "ftag": sc["fulltime"]["away"] if done else None,
                })
    return pd.DataFrame(rows, columns=COLS)


def split_cups(cups: pd.DataFrame):
    """Return (finished cup matches, upcoming cup fixtures)."""
    if cups.empty:
        return cups, cups
    for c in ["hthg", "htag", "fthg", "ftag"]:
        cups[c] = pd.to_numeric(cups[c], errors="coerce")
    done = cups.dropna(subset=["hthg", "htag", "fthg", "ftag"])
    now = (pd.Timestamp(os.getenv("H2_TODAY")) if os.getenv("H2_TODAY")
           else pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None))
    todo = cups[cups["fthg"].isna() & (cups["date"] >= now - pd.Timedelta(hours=3))]
    return done, todo

# ======================================================================
# names
# ======================================================================
"""Map cup-source team names onto football-data.co.uk league names."""

import csv
import difflib
import re
import unicodedata


ALIAS_FILE = C.DATA / "aliases.csv"
_STOP = {"fc", "afc", "cf", "sc", "ac", "as", "ssc", "sv", "vfb", "vfl", "tsg",
         "fsv", "club", "calcio", "sk", "fk", "cd", "ud", "rc", "rcd", "sd",
         "real", "royal", "kv", "krc", "the", "de", "1", "04", "05", "1899",
         "1909", "1846", "1848", "hotspur", "albion", "wanderers", "united",
         "city", "town", "athletic", "olympique", "sporting", "stade", "losc"}


def norm(s: str, drop_stop=True) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("&", " "))
    toks = s.split()
    if drop_stop:
        kept = [t for t in toks if t not in _STOP]
        toks = kept or toks
    return " ".join(toks)


def load_aliases() -> dict[str, str]:
    out = {}
    if ALIAS_FILE.exists():
        with ALIAS_FILE.open(encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and not row[0].startswith("#"):
                    out[norm(row[0], False)] = row[1].strip()
    return out


class Matcher:
    def __init__(self, league_names):
        self.names = sorted(set(league_names))
        self.by_norm = {}
        for n in self.names:
            self.by_norm.setdefault(norm(n), n)
        self.alias = load_aliases()
        self.unmatched: set[str] = set()
        self._cache: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        res = self._match(name)
        self._cache[name] = res
        return res

    def _match(self, name):
        if name in self.names:
            return name
        a = self.alias.get(norm(name, False))
        if a:
            return a
        n = norm(name)
        if n in self.by_norm:
            return self.by_norm[n]
        close = difflib.get_close_matches(n, self.by_norm.keys(), n=1, cutoff=0.86)
        if close:
            return self.by_norm[close[0]]
        # containment, e.g. "borussia dortmund" vs "dortmund"
        hits = [k for k in self.by_norm if len(k) >= 5 and (k in n.split() or k == n
                or (len(n) >= 5 and n in k))]
        if len(hits) == 1:
            return self.by_norm[hits[0]]
        self.unmatched.add(name)
        return name   # keeps its own (cup-only) rating

# ======================================================================
# model
# ======================================================================
"""Second-half goal model.

1. Walk forward through every historical match in date order. Before each
   match, record a pre-match feature vector (so every row is genuinely
   out-of-sample); afterwards, update the teams' ratings.
2. Ratings: exponentially-decayed, shrunk-to-league-average multipliers for
   2nd-half attack and defence, venue-specific and venue-blind, measured
   relative to the competition's own 2nd-half scoring rate.
3. Expected 2H goals  lam = Bh*att(H)*def(A) + Ba*att(A)*def(H);
   raw P(>=1 goal) = 1 - exp(-lam).
4. A logistic layer calibrates the raw probability using the team and league
   2H-goal hit rates, trained only on earlier seasons and tested on later ones.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


FEATURES = ["f_pois", "f_league", "f_hit_h", "f_hit_a", "f_lam_ratio"]
_EPS = 1e-4


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class _EW:
    """Decayed weighted mean with prior shrinkage."""
    __slots__ = ("s", "w")

    def __init__(self):
        self.s = 0.0; self.w = 0.0

    def add(self, x, decay):
        self.s = self.s * decay + x; self.w = self.w * decay + 1.0

    def shrink(self, f):
        self.s *= f; self.w *= f

    def val(self, prior, k):
        return (self.s + k * prior) / (self.w + k)


@dataclass
class Team:
    att_h: _EW = field(default_factory=_EW); def_h: _EW = field(default_factory=_EW)
    att_a: _EW = field(default_factory=_EW); def_a: _EW = field(default_factory=_EW)
    att: _EW = field(default_factory=_EW);   dfn: _EW = field(default_factory=_EW)
    hit: _EW = field(default_factory=_EW)
    n: int = 0
    league: str | None = None


@dataclass
class League:
    bh: float = 0.75   # home 2H goals per match (starting prior)
    ba: float = 0.60
    hit: float = 0.78
    n: int = 0


class Engine:
    def __init__(self):
        self.teams: dict[str, Team] = defaultdict(Team)
        self.leagues: dict[str, League] = defaultdict(League)
        self.td = 0.5 ** (1 / C.TEAM_HALFLIFE)
        self.ld = 1 - 0.5 ** (1 / C.LEAGUE_HALFLIFE)   # EWMA step

    # -- baselines --------------------------------------------------------
    def _global(self):
        ls = [l for l in self.leagues.values() if l.n > 50] or [League()]
        return (np.mean([l.bh for l in ls]), np.mean([l.ba for l in ls]),
                np.mean([l.hit for l in ls]))

    def baseline(self, comp, kind, home, away):
        if kind == "league":
            l = self.leagues[comp]
            return l.bh, l.ba, l.hit
        g = self._global()
        vals = []
        for t in (home, away):
            lg = self.teams[t].league if t in self.teams else None
            if lg and self.leagues[lg].n > 50:
                l = self.leagues[lg]; vals.append((l.bh, l.ba, l.hit))
            else:
                vals.append(g)
        return tuple(float(np.mean([v[i] for v in vals])) for i in range(3))

    # -- features ---------------------------------------------------------
    def features(self, comp, kind, home, away):
        k = C.SHRINK_K
        bh, ba, bhit = self.baseline(comp, kind, home, away)
        H, A = self.teams[home], self.teams[away]
        att_H = 0.5 * H.att_h.val(1, k) + 0.5 * H.att.val(1, k)
        def_H = 0.5 * H.def_h.val(1, k) + 0.5 * H.dfn.val(1, k)
        att_A = 0.5 * A.att_a.val(1, k) + 0.5 * A.att.val(1, k)
        def_A = 0.5 * A.def_a.val(1, k) + 0.5 * A.dfn.val(1, k)
        lam = bh * att_H * def_A + ba * att_A * def_H
        p = 1 - math.exp(-lam)
        return {
            "lam": lam, "p_pois": p, "league_hit": bhit,
            "hit_h": H.hit.val(bhit, k), "hit_a": A.hit.val(bhit, k),
            "n_h": H.n, "n_a": A.n,
            "f_pois": float(_logit(p)), "f_league": float(_logit(bhit)),
            "f_hit_h": float(_logit(H.hit.val(bhit, k))),
            "f_hit_a": float(_logit(A.hit.val(bhit, k))),
            "f_lam_ratio": math.log(lam / (bh + ba)),
        }

    # -- update -----------------------------------------------------------
    def update(self, comp, kind, home, away, gh, ga):
        bh, ba, _ = self.baseline(comp, kind, home, away)
        d = self.td
        for name in (home, away):
            t = self.teams[name]
            if kind == "league" and t.league and t.league != comp:
                # promoted / relegated: ratings were relative to another league
                for ew in (t.att_h, t.def_h, t.att_a, t.def_a, t.att, t.dfn, t.hit):
                    ew.shrink(0.35)
            if kind == "league":
                t.league = comp
        H, A = self.teams[home], self.teams[away]
        rh, ra = gh / bh, ga / ba
        H.att_h.add(rh, d); H.att.add(rh, d); H.def_h.add(ra, d); H.dfn.add(ra, d)
        A.att_a.add(ra, d); A.att.add(ra, d); A.def_a.add(rh, d); A.dfn.add(rh, d)
        hit = 1.0 if gh + ga > 0 else 0.0
        H.hit.add(hit, d); A.hit.add(hit, d)
        H.n += 1; A.n += 1
        if kind == "league":
            l, s = self.leagues[comp], self.ld
            if l.n == 0:
                pass
            l.bh += s * (gh - l.bh); l.ba += s * (ga - l.ba)
            l.hit += s * (hit - l.hit); l.n += 1


def walk_forward(matches: pd.DataFrame):
    """Return (engine with final state, DataFrame of out-of-sample rows)."""
    m = matches.sort_values("date", kind="stable").reset_index(drop=True)
    eng, rows = Engine(), []
    # seed league baselines from first 100 matches so early rows aren't junk
    for comp, g in m[m.kind == "league"].groupby("comp"):
        g = g.head(100)
        l = eng.leagues[comp]
        l.bh = float((g.fthg - g.hthg).mean()); l.ba = float((g.ftag - g.htag).mean())
        l.hit = float(((g.fthg + g.ftag - g.hthg - g.htag) > 0).mean())
    for r in m.itertuples(index=False):
        f = eng.features(r.comp, r.kind, r.home, r.away)
        gh, ga = max(r.fthg - r.hthg, 0), max(r.ftag - r.htag, 0)
        f.update(date=r.date, comp=r.comp, comp_name=r.comp_name, kind=r.kind,
                 home=r.home, away=r.away, g2h=gh + ga, y=int(gh + ga > 0))
        rows.append(f)
        eng.update(r.comp, r.kind, r.home, r.away, gh, ga)
    return eng, pd.DataFrame(rows)


@dataclass
class Calibrator:
    lr: LogisticRegression | None = None

    def fit(self, df):
        self.lr = LogisticRegression(C=1.0, max_iter=1000)
        self.lr.fit(df[FEATURES].values, df["y"].values)
        return self

    def predict(self, df):
        return self.lr.predict_proba(df[FEATURES].values)[:, 1]


def eligible(df):
    return df[(df.n_h >= C.MIN_TEAM_MATCHES) & (df.n_a >= C.MIN_TEAM_MATCHES)]


def backtest(oos: pd.DataFrame, test_seasons=2):
    """Train calibration on older seasons, evaluate on the most recent ones."""
    df = eligible(oos).copy()
    # skip the first season entirely (ratings still warming up)
    first = df.date.min() + pd.Timedelta(days=330)
    df = df[df.date >= first]
    last = df.date.max()
    yr = last.year if last.month >= 7 else last.year - 1
    cutoff = pd.Timestamp(yr - test_seasons + 1, 7, 1)
    train, test = df[df.date < cutoff], df[df.date >= cutoff].copy()
    cal = Calibrator().fit(train)
    test["p"] = cal.predict(test)
    base = train.y.mean()

    def brier(p, y): return float(np.mean((p - y) ** 2))

    def ll(p, y):
        p = np.clip(p, _EPS, 1 - _EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    summary = {
        "train_matches": int(len(train)), "test_matches": int(len(test)),
        "test_from": cutoff.date().isoformat(),
        "test_to": test.date.max().date().isoformat() if len(test) else None,
        "base_rate": float(test.y.mean()),
        "brier_model": brier(test.p, test.y), "brier_baseline": brier(np.full(len(test), base), test.y),
        "logloss_model": ll(test.p, test.y), "logloss_baseline": ll(np.full(len(test), base), test.y),
    }
    thr = []
    for t in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]:
        s = test[test.p >= t]
        thr.append({"threshold": t, "picks": int(len(s)),
                    "per_week": round(len(s) / max((test.date.max() - test.date.min()).days / 7, 1), 1),
                    "hit_rate": float(s.y.mean()) if len(s) else None,
                    "avg_pred": float(s.p.mean()) if len(s) else None,
                    "fair_odds_hit": round(1 / s.y.mean(), 3) if len(s) and s.y.mean() > 0 else None})
    bins = pd.cut(test.p, [0, .6, .7, .75, .8, .85, .88, .9, .92, 1])
    calib = (test.groupby(bins, observed=True)
                 .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
                 .reset_index().rename(columns={"p": "band"}))
    calib["band"] = calib["band"].astype(str)
    picks = test[test.p >= C.PICK_THRESHOLD]
    by_league = (picks.groupby("comp_name")
                      .agg(picks=("y", "size"), hit_rate=("y", "mean"))
                      .sort_values("picks", ascending=False).reset_index())
    return summary, pd.DataFrame(thr), calib, by_league, test

# ======================================================================
# run
# ======================================================================
"""Daily pipeline:  python -m h2goals.run"""

import html
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests


log = logging.getLogger("h2goals")
LOG_FILE = C.DATA / "picks_log.csv"


def now():
    o = os.getenv("H2_TODAY")
    return pd.Timestamp(o) if o else pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)


def gather():
    hist = F.fetch_league_history()
    fx = F.fetch_league_fixtures()
    cups = pd.concat([F.fetch_fdorg_cups(), F.fetch_apif_cups()], ignore_index=True)
    match = Matcher(hist["home"].tolist() + hist["away"].tolist())
    if not cups.empty:
        cups["home"] = cups["home"].map(match); cups["away"] = cups["away"].map(match)
    cup_done, cup_fx = F.split_cups(cups)
    hist = pd.concat([hist, cup_done], ignore_index=True)
    hist = hist.drop_duplicates(subset=["date", "home", "away"])
    fixtures = pd.concat([fx, cup_fx], ignore_index=True)
    for df in (hist, fixtures):
        df["date"] = pd.to_datetime(df["date"])
        for c in ["hthg", "htag", "fthg", "ftag"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return hist, fixtures, match.unmatched


def score_fixtures(eng, cal, fixtures, hist):
    t0 = now()
    fx = fixtures[(fixtures.date >= t0 - pd.Timedelta(hours=3)) &
                  (fixtures.date <= t0 + pd.Timedelta(days=C.DAYS_AHEAD))].copy()
    if fx.empty:
        return fx
    played = set(zip(hist.date.dt.date, hist.home, hist.away))
    fx = fx[[(d.date(), h, a) not in played for d, h, a in zip(fx.date, fx.home, fx.away)]]
    rows = []
    for r in fx.itertuples(index=False):
        f = eng.features(r.comp, r.kind, r.home, r.away)
        f.update(date=r.date, comp=r.comp, comp_name=r.comp_name, kind=r.kind,
                 home=r.home, away=r.away)
        rows.append(f)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p"] = cal.predict(out)
    out["fair_odds"] = (1 / out["p"]).round(3)
    out["rated"] = (out.n_h >= C.MIN_TEAM_MATCHES) & (out.n_a >= C.MIN_TEAM_MATCHES)
    out["pick"] = out.rated & (out.p >= C.PICK_THRESHOLD)
    return out.sort_values(["pick", "p"], ascending=False)


def update_log(scored, hist):
    cols = ["date", "comp_name", "home", "away", "p", "fair_odds", "logged",
            "g2h", "result"]
    logdf = (pd.read_csv(LOG_FILE, parse_dates=["date"], dtype={"result": "object"})
             if LOG_FILE.exists() else pd.DataFrame(columns=cols))
    logdf["result"] = logdf["result"].astype("object")
    if not scored.empty:
        new = scored[scored.pick][["date", "comp_name", "home", "away", "p", "fair_odds"]].copy()
        new["logged"] = now().date().isoformat()
        key = set(zip(logdf.date.astype(str).str[:10], logdf.home, logdf.away))
        new = new[[(str(d)[:10], h, a) not in key for d, h, a in zip(new.date, new.home, new.away)]]
        logdf = pd.concat([logdf, new], ignore_index=True)
    # settle
    h = hist.assign(day=hist.date.dt.date.astype(str),
                    g=(hist.fthg - hist.hthg + hist.ftag - hist.htag))
    res = {(d, a, b): g for d, a, b, g in zip(h.day, h.home, h.away, h.g)}
    for i, r in logdf.iterrows():
        if pd.isna(r.get("result")) or r.get("result") == "":
            g = res.get((str(r.date)[:10], r.home, r.away))
            if g is not None:
                logdf.at[i, "g2h"] = int(g)
                logdf.at[i, "result"] = "WIN" if g > 0 else "LOSS"
    logdf["g2h"] = pd.to_numeric(logdf["g2h"], errors="coerce").astype("Int64")
    logdf = logdf[cols].sort_values("date")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logdf.to_csv(LOG_FILE, index=False)
    return logdf


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1b1d1f;--mute:#6b7075;--line:#e3e3df;
--good:#0f7b4c;--bad:#b3261e;--accent:#1f5fbf;--chip:#eef3fb}
@media (prefers-color-scheme:dark){:root{--bg:#121416;--card:#1b1e21;--ink:#e8e9ea;
--mute:#9aa0a6;--line:#2c3034;--good:#4cc38a;--bad:#f28b82;--accent:#8ab4f8;--chip:#1f2a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.sub{color:var(--mute);margin:0 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mute);font-size:13px}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute);font-weight:600}
tr:last-child td{border-bottom:0}td.n{text-align:right}th.n{text-align:right}
.pick td{background:var(--chip)}.win{color:var(--good);font-weight:600}.loss{color:var(--bad);font-weight:600}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);color:var(--accent);font-size:12px}
.note{color:var(--mute);font-size:13px;margin-top:8px}
"""


def _table(df, cols, fmt=None, row_cls=None):
    fmt = fmt or {}
    head = "".join(f'<th class="{"n" if c in fmt else ""}">{html.escape(l)}</th>' for c, l in cols)
    body = []
    for _, r in df.iterrows():
        cls = row_cls(r) if row_cls else ""
        tds = []
        for c, _l in cols:
            v = r[c]
            if c in fmt and pd.notna(v):
                tds.append(f'<td class="n">{fmt[c](v)}</td>')
            elif c == "result" and isinstance(v, str):
                tds.append(f'<td class="{v.lower()}">{v}</td>')
            else:
                tds.append(f"<td>{'' if pd.isna(v) else html.escape(str(v))}</td>")
        body.append(f'<tr class="{cls}">{"".join(tds)}</tr>')
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


pct = lambda v: f"{v*100:.1f}%"


def write_report(scored, summary, thr, calib, by_league, logdf, unmatched, n_hist):
    C.DOCS.mkdir(parents=True, exist_ok=True)
    picks = scored[scored.pick] if not scored.empty else scored
    watch = scored[~scored.pick & scored.rated].head(15) if not scored.empty else scored
    if not scored.empty:
        scored.assign(date=scored.date.dt.strftime("%Y-%m-%d %H:%M"))[
            ["date", "comp_name", "home", "away", "p", "fair_odds", "lam", "hit_h", "hit_a",
             "n_h", "n_a", "pick"]].to_csv(C.DOCS / "fixtures_scored.csv", index=False)
    settled = logdf[logdf.result.isin(["WIN", "LOSS"])]
    live_hit = (settled.result == "WIN").mean() if len(settled) else None

    def fx_table(df):
        d = df.assign(when=df.date.dt.strftime("%a %d %b %H:%M"))
        return _table(d, [("when", "Kick-off (UK)"), ("comp_name", "Competition"), ("home", "Home"),
                          ("away", "Away"), ("p", "P(2H goal)"), ("fair_odds", "Fair odds"),
                          ("lam", "Exp. 2H goals"), ("hit_h", "Home 2H-goal rate"), ("hit_a", "Away 2H-goal rate")],
                      {"p": pct, "fair_odds": lambda v: f"{v:.2f}", "lam": lambda v: f"{v:.2f}",
                       "hit_h": pct, "hit_a": pct})

    t = next((x for _, x in thr.iterrows() if abs(x.threshold - C.PICK_THRESHOLD) < 1e-9), None)
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>2nd-Half Goal Finder</title>
<style>{CSS}</style></head><body><main>
<h1>2nd-Half Goal Finder</h1>
<p class="sub">Updated {now():%a %d %b %Y %H:%M} · next {C.DAYS_AHEAD} days · threshold {pct(C.PICK_THRESHOLD)} · model built on {n_hist:,} historical matches</p>
<div class="kpis">
<div class="kpi"><b>{len(picks)}</b><span>picks in window</span></div>
<div class="kpi"><b>{pct(t.hit_rate) if t is not None and t.hit_rate else '–'}</b><span>backtest hit rate at threshold ({int(t.picks) if t is not None else 0} matches)</span></div>
<div class="kpi"><b>{pct(summary['base_rate'])}</b><span>all-match base rate (test period)</span></div>
<div class="kpi"><b>{pct(live_hit) if live_hit is not None else '–'}</b><span>live record ({len(settled)} settled)</span></div>
</div>
<h2>Picks</h2>""",
             fx_table(picks) if len(picks) else '<p class="note">No fixture clears the threshold in this window.</p>',
             '<p class="note">Only back a pick when the available price is above its fair odds. Over 0.5 2H goals is usually priced tight; the edge, if any, is in the gap.</p>',
             "<h2>Next best (below threshold)</h2>",
             fx_table(watch) if len(watch) else '<p class="note">None.</p>',
             "<h2>Backtest — out-of-sample</h2>",
             f'<p class="note">Calibration trained on matches before {summary["test_from"]}, tested on {summary["test_matches"]:,} later matches. '
             f'Brier {summary["brier_model"]:.4f} vs {summary["brier_baseline"]:.4f} for a flat base rate; '
             f'log-loss {summary["logloss_model"]:.4f} vs {summary["logloss_baseline"]:.4f}.</p>',
             _table(thr, [("threshold", "Threshold"), ("picks", "Matches"), ("per_week", "Per week"),
                          ("hit_rate", "Hit rate"), ("avg_pred", "Avg predicted"), ("fair_odds_hit", "Break-even odds")],
                    {"threshold": pct, "hit_rate": pct, "avg_pred": pct, "fair_odds_hit": lambda v: f"{v:.3f}",
                     "per_week": lambda v: f"{v:.1f}", "picks": lambda v: f"{int(v):,}"}),
             "<h2>Calibration (predicted vs actual)</h2>",
             _table(calib, [("band", "Predicted band"), ("n", "Matches"), ("predicted", "Avg predicted"), ("actual", "Actual")],
                    {"predicted": pct, "actual": pct, "n": lambda v: f"{int(v):,}"}),
             "<h2>Backtest picks by competition</h2>",
             _table(by_league.head(25), [("comp_name", "Competition"), ("picks", "Picks"), ("hit_rate", "Hit rate")],
                    {"hit_rate": pct, "picks": lambda v: f"{int(v):,}"}),
             "<h2>Live pick log</h2>",
             _table(logdf.sort_values("date", ascending=False).head(60).assign(
                 date=lambda d: pd.to_datetime(d.date).dt.strftime("%d %b %H:%M")),
                 [("date", "Kick-off"), ("comp_name", "Competition"), ("home", "Home"), ("away", "Away"),
                  ("p", "P"), ("g2h", "2H goals"), ("result", "Result")], {"p": pct})
             if len(logdf) else '<p class="note">Picks will be logged and settled here automatically.</p>']
    if unmatched:
        parts.append(f'<p class="note">{len(unmatched)} cup team names not matched to league data (rated from cup games only) — see unmatched.txt.</p>')
    parts.append("</main></body></html>")
    (C.DOCS / "index.html").write_text("".join(parts), encoding="utf-8")
    (C.DOCS / "backtest.json").write_text(json.dumps(
        {"summary": summary, "thresholds": thr.to_dict("records")}, indent=2, default=str))
    (C.DOCS / "unmatched.txt").write_text("\n".join(sorted(unmatched)))


def telegram(picks):
    if not (C.TELEGRAM_TOKEN and C.TELEGRAM_CHAT_ID) or picks.empty:
        return
    lines = ["⚽ 2H goal picks"] + [
        f"{r.date:%a %H:%M} {r.home} v {r.away} ({r.comp_name}) — {r.p*100:.0f}%, fair {r.fair_odds:.2f}"
        for r in picks.itertuples()]
    requests.post(f"https://api.telegram.org/bot{C.TELEGRAM_TOKEN}/sendMessage",
                  json={"chat_id": C.TELEGRAM_CHAT_ID, "text": "\n".join(lines)}, timeout=20)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    hist, fixtures, unmatched = gather()
    if hist.empty:
        raise SystemExit("No historical data downloaded — check network / sources.")
    log.info("history: %d matches, fixtures: %d", len(hist), len(fixtures))
    eng, oos = walk_forward(hist)
    summary, thr, calib, by_league, _ = backtest(oos)
    log.info("backtest: %s", summary)
    live = eligible(oos)
    live = live[live.date >= live.date.min() + pd.Timedelta(days=330)]
    cal = Calibrator().fit(live)
    scored = score_fixtures(eng, cal, fixtures, hist)
    logdf = update_log(scored, hist)
    write_report(scored, summary, thr, calib, by_league, logdf, unmatched, len(hist))
    picks = scored[scored.pick] if not scored.empty else scored
    log.info("picks: %d", len(picks))
    telegram(picks)



# ======================================================================
# built-in team-name aliases (extra ones can go in data/aliases.csv)
# ======================================================================
BUILTIN_ALIASES = """# cup-source name,football-data.co.uk name   (add a line for anything in docs/unmatched.txt)
Manchester United,Man United
Manchester United FC,Man United
Manchester City,Man City
Manchester City FC,Man City
Nottingham Forest,Nott'm Forest
Wolverhampton Wanderers,Wolves
West Bromwich Albion,West Brom
Sheffield Wednesday,Sheffield Weds
Queens Park Rangers,QPR
Bayern München,Bayern Munich
FC Bayern München,Bayern Munich
Bayern Munich,Bayern Munich
Borussia Mönchengladbach,M'gladbach
Borussia Monchengladbach,M'gladbach
Eintracht Frankfurt,Ein Frankfurt
Bayer Leverkusen,Leverkusen
Bayer 04 Leverkusen,Leverkusen
Paris Saint-Germain,Paris SG
Paris Saint Germain,Paris SG
Paris Saint-Germain FC,Paris SG
Atletico Madrid,Ath Madrid
Atlético Madrid,Ath Madrid
Club Atlético de Madrid,Ath Madrid
Athletic Club,Ath Bilbao
Athletic Bilbao,Ath Bilbao
Real Sociedad,Sociedad
Real Betis,Betis
Celta Vigo,Celta
RCD Espanyol de Barcelona,Espanol
Espanyol,Espanol
Rayo Vallecano,Vallecano
Sporting CP,Sp Lisbon
Sporting Clube de Portugal,Sp Lisbon
Sporting Braga,Sp Braga
SC Braga,Sp Braga
Inter,Inter
FC Internazionale Milano,Inter
Inter Milan,Inter
AC Milan,Milan
Union Saint-Gilloise,St. Gilloise
Royale Union Saint-Gilloise,St. Gilloise
Olympiacos,Olympiakos
Olympiakos Piraeus,Olympiakos
AEK Athens,AEK
PSV,PSV Eindhoven
"""
_load_file_aliases = load_aliases


def load_aliases():
    out = {}
    for line in BUILTIN_ALIASES.splitlines():
        row = line.split(",")
        if len(row) >= 2 and not row[0].startswith("#"):
            out[norm(row[0], False)] = row[1].strip()
    out.update(_load_file_aliases())
    return out


if __name__ == "__main__":
    main()
