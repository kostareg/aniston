# Aniston

ESPN fantasy football analysis and trade-edge engine for **DU FANTASY** (2026, redraft).

The core idea: other managers judge players by what the ESPN app shows them — ESPN's projections, season points,
recent games and what the player cost in the draft (**perceived value**). Aniston values players by a consensus of
projection sources, measured as marginal starting-lineup points on *your* roster and converted into
**championship probability** by simulating the rest of the season (**true value**). Trades where the gap between
the two is large, in your favor, are the edge.

## Setup

```fish
nix develop            # Python 3.12 + uv, with the native libs numpy/polars/duckdb wheels need on NixOS
uv sync
cp .env.example .env   # fill ESPN_S2 and SWID (browser cookies from espn.com); LEAGUE_ID is in .env already
```

## Use

```fish
uv run aniston refresh     # pull ESPN, Sleeper (RotoWire projections), nflverse into data/aniston.duckdb
uv run aniston analyze     # projections, season sim, waivers, trade search (~30s); stores results
uv run aniston trades -n 10 [--partner name]
uv run aniston sim
uv run aniston backtest    # projection accuracy per source on completed weeks
uv run streamlit run app/Home.py
```

Automatic refresh (Tue after MNF, Wed after waivers, Sun before kickoff):

```fish
mkdir -p ~/.config/systemd/user
cp deploy/aniston-refresh.* ~/.config/systemd/user/
systemctl --user daemon-reload; and systemctl --user enable --now aniston-refresh.timer
```

## How it works

| Stage | Module | What it does |
|---|---|---|
| Ingest | `ingest/espn.py` | League settings + scoring, teams, schedule, live scores, rosters, **weekly ESPN projections for every week**, ADP, draft, trades |
| | `ingest/sleeper.py` | Sleeper's weekly projections (RotoWire) as stat lines, scored with your league's settings |
| | `ingest/nflverse.py` | Weekly stats, usage (targets, carries, shares), **expected fantasy points** (xFP), ID map, schedules |
| Model | `model/projections.py` | Per player-week ensemble: ESPN + Sleeper + usage model (xFP shrunk toward projections). Weights in `.env` |
| | `model/lineup.py` | Optimal lineup (QB/2RB/2WR/TE/FLEX/D/K), vectorized over sims and weeks |
| | `model/variance.py` | Weekly volatility by position fitted on last season |
| | `model/signals.py` | Luck (points vs xFP), usage trends, ESPN-vs-consensus gap, schedule strength (ROS + playoff weeks), buy-low/sell-high |
| Sim | `sim/season.py` | 5,000 seasons: per-player projection error, injury spells, team drift, weekly score noise → standings → seeding (points tiebreak) → 6-team bracket. Common random numbers make trade deltas precise |
| Trades | `trade/perceived.py` | Perceived value = ESPN ROS, season PPG, last-3 PPG and draft cost, scaled to games left |
| | `trade/roster.py` | Roster limits, IR, drops (partners drop by perceived value), best free-agent pickup |
| | `trade/acceptance.py` | P(accept) from their **ESPN-visible** lineup gain + perceived value; P(no veto) from how lopsided it looks |
| | `trade/engine.py` | 1-1, 2-1, 1-2, 2-2 search with every team → fast proxies → full simulation of the top 60 |
| | `trade/explain.py` | Why it works for you, why it looks good to them, player-level signals |

Ranking: **EV = Δ P(title) × P(accept) × P(no veto)**.

## Honest limitations

- The acceptance and veto models are hand-set logistic curves (`ACC_*`, `VETO_*` in `.env`); there are too few trades
  in league history to fit them. Treat P(accept) as a ranking signal, not a calibrated probability.
- Playoff odds are more confident than ESPN's because they use roster strength, not just record. The knobs
  (`PLAYER_SHOCK_SD`, `TEAM_SHOCK_SD`, `INJURY_RATE`) are documented in `config.py`.
- Sleeper's projection endpoint and ESPN's raw views are unofficial; each source fails independently.
- Ensemble weights come from a 4-week backtest (Sleeper was most accurate); rerun `aniston backtest` as weeks accrue.
