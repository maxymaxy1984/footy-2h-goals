# 2nd-Half Goal Finder

Scans European leagues and cups every day and flags matches with a very high
probability of **at least one goal after half-time** (Over 0.5 2H goals).
Results publish to a web page that updates itself each morning, and every
pick is logged and settled automatically so you build a real track record.

## What it covers

| Source | Competitions | Key needed |
|---|---|---|
| football-data.co.uk | 22 leagues: England (5 tiers), Scotland (4), Germany (2), Italy (2), Spain (2), France (2), Netherlands, Belgium, Portugal, Turkey, Greece | none |
| football-data.org | Champions League | free key |
| API-Football | Europa League, Conference League, FA Cup, League Cup, DFB-Pokal, Coppa Italia, Copa del Rey, Coupe de France, KNVB Beker, Taça de Portugal, Belgian Cup, Turkish Cup, Scottish Cup | free key (100 calls/day) |

Leagues work with no keys at all. Cups switch on when you add the keys.
History: 8 seasons for leagues (~55,000 matches), 3–4 for cups.

## How it decides

1. **Team 2nd-half ratings.** For every team: 2H goals scored and conceded, home
   and away, relative to its competition's 2H scoring rate. Recent matches weigh
   more (half-life 19 games); small samples are pulled towards league average;
   promoted/relegated teams are partially reset.
2. **Expected 2H goals** = home attack × away defence × league home rate +
   away attack × home defence × league away rate → Poisson P(≥1 goal).
3. **Calibration.** A logistic layer combines that with each team's and league's
   2H-goal hit rate. It is trained on older seasons only.
4. **Honest backtest.** Every historical prediction is made using only data
   available before kick-off; the report shows hit rate, matches per week and
   predicted-vs-actual for the last two seasons the model never trained on.
5. **Picks** = calibrated probability ≥ 86% (change with `H2_THRESHOLD`) and
   both teams have ≥ 8 rated matches. Cup ties between leagues use each club's
   domestic rating against the average of the two leagues.

## Set up (about 10 minutes)

1. Create a new **private** GitHub repo and upload this folder's contents.
2. Repo **Settings → Pages**: Source "Deploy from a branch", branch `main`,
   folder `/docs`. Your report lives at the URL Pages shows you.
   (Pages on a private repo needs a paid GitHub plan; on a free plan make the
   repo public or just open `docs/index.html` from the repo.)
3. Repo **Settings → Secrets and variables → Actions → New repository secret** (optional):
   - `FOOTBALL_DATA_ORG_KEY` — register at football-data.org (free) for the Champions League
   - `API_FOOTBALL_KEY` — register at dashboard.api-football.com (free) for the other cups
   - `TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` — daily picks pushed to your phone
4. **Actions** tab → "Daily 2H goal scan" → **Run workflow** for the first run
   (downloads history, ~3–5 min). After that it runs itself at 06:17 UTC daily.

Run locally instead: `pip install -r requirements.txt && python -m h2goals.run`,
then open `docs/index.html`.

## Files it produces

- `docs/index.html` — picks, next-best list, backtest, calibration, live log
- `docs/fixtures_scored.csv` — every fixture in the window with its probability
- `data/picks_log.csv` — all picks with results filled in automatically
- `docs/unmatched.txt` — cup team names not matched to league data; add a line
  to `data/aliases.csv` to fix any that matter

## Using it sensibly

- Across Europe about 78–80% of matches have a 2H goal, and bookmakers and
  Betfair price Over 0.5 2H accordingly (typically around 1.20–1.30 pre-match).
  The useful number is the gap between the model's **fair odds** and the price
  on offer — only back where the price is longer than fair odds.
- Check the backtest table on the page before trusting a threshold: higher
  thresholds give higher hit rates but far fewer matches.
- Team news, rotation before European nights and dead rubbers are not in the
  data. A quick look before kick-off still matters.
- Free API tiers can change which seasons they serve; if a cup shows no data,
  the run log in the Actions tab says why.

## Settings (`h2goals/config.py` or environment variables)

`H2_THRESHOLD` (0.86) · `H2_DAYS_AHEAD` (3) · `H2_SEASONS_BACK` (8) ·
`TEAM_HALFLIFE` · `SHRINK_K` · `MIN_TEAM_MATCHES` · league and cup lists.

`tests/make_synthetic.py` builds fake data in the same format so the pipeline
can be tested offline: `python tests/make_synthetic.py . 2026-10-24` then
`H2_OFFLINE=1 H2_TODAY=2026-10-24 python -m h2goals.run`
(delete `data/raw` afterwards).
