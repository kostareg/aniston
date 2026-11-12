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
uv run aniston reload      # refresh + analyze in one go
uv run aniston trades -n 10 [--partner name]
uv run aniston target "player name"   # best packages that land a specific player
uv run aniston sim
uv run aniston backtest    # projection accuracy per source on completed weeks
uv run aniston digest [--dry-run]     # what changed since the last digest; sent to NOTIFY_URL when set
uv run aniston offers                 # trades you proposed; `aniston offer-status 3 accepted` records the outcome
uv run streamlit run app/Home.py
```

Set `NOTIFY_URL` in `.env` to an ntfy topic URL or a Discord webhook to get the digest pushed after every
scheduled refresh. Log the offers you actually send (Trade Finder / Trade Analyzer, then the Offers page for the
outcome): they are the only data the acceptance model can be checked against.

## Self-hosting (NixOS)

The dashboard runs as a systemd user service on port 1313, and a timer refreshes data and models and sends
the digest (Tue after MNF, Wed after waivers, Thu before kickoff, Sun after inactives; the times assume the
server's clock is US Eastern); both are optional. Without the timer, reload by hand with `aniston reload` or the
Home page's refresh button, which runs that command as its own process so the trade search uses every core. Clone to `~/Projects/aniston`, copy `.env`, then:

```fish
mkdir -p ~/.config/systemd/user
cp deploy/aniston-* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aniston-dashboard.service aniston-refresh.timer
journalctl --user -u aniston-dashboard -f    # logs; first start builds the dev shell
```

User services stop at logout unless lingering is on. In `configuration.nix`:
`users.users.<you>.linger = true;`

The dashboard listens on `127.0.0.1:1313` because it has no login and shows your trade targets. Reach it through
Cloudflare Tunnel + Access (or a reverse proxy with auth). To expose it directly instead, set
`STREAMLIT_SERVER_ADDRESS=0.0.0.0` in `deploy/aniston-dashboard.service` and open the port:
`networking.firewall.allowedTCPPorts = [ 1313 ];`

## How it works

| Stage | Module | What it does |
|---|---|---|
| Ingest | `ingest/espn.py` | League settings + scoring, teams, schedule, live scores, rosters, **weekly ESPN projections for every week**, ADP, draft, trades |
| | `ingest/sleeper.py` | Sleeper's weekly projections (RotoWire) as stat lines, scored with your league's settings |
| | `ingest/nflverse.py` | Weekly stats, usage (targets, carries, shares), **expected fantasy points** (xFP), ID map, schedules with kickoff times |
| Model | `model/projections.py` | Per player-week ensemble: ESPN + Sleeper + usage model (xFP shrunk toward projections). Weights in `.env`. The current week counts as a future week until its first kickoff. Free agents are picked by the higher of ESPN's and Sleeper's ROS, plus each week's best streamers |
| | `model/lineup.py` | Optimal lineup (QB/2RB/2WR/TE/FLEX/D/K), vectorized over sims and weeks |
| | `model/variance.py` | Weekly volatility by position fitted on last season |
| | `model/signals.py` | Luck (points vs xFP), usage trends, ESPN-vs-consensus gap, schedule strength (ROS + playoff weeks), buy-low/sell-high |
| | `model/startsit.py` | This week's lineup and one-for-one swaps scored by win probability against the opponent (underdogs want volatility, favourites don't), using per-player volatility |
| Sim | `sim/season.py` | 5,000 seasons: per-player projection error, injury spells, team drift, weekly score noise → standings → seeding (points tiebreak) → 8-team bracket. Common random numbers make trade deltas precise. Finished and in-progress weeks, playoff rounds included, come from stored and live scores |
| Trades | `trade/perceived.py` | Perceived value = ESPN ROS, season PPG, last-3 PPG and draft cost, scaled to games left |
| | `trade/roster.py` | Roster limits, IR, drops (partners drop by perceived value), best free-agent pickup |
| | `trade/acceptance.py` | P(accept) from their **ESPN-visible** lineup gain + perceived value, weighted by their situation (bubble teams want points now, eliminated teams rarely answer); P(no veto) from how lopsided it looks |
| | `trade/engine.py` | 1-1, 2-1, 1-2, 2-2 search with every team (or for one target player) → fast proxies → full simulation of the top 60 |
| | `trade/explain.py` | Why it works for you, why it looks good to them, player-level signals |
| After | `analyze.py` | Waivers with playoff-week gain, streamers (QB/K/D/ST), blocks against this week's opponent; skips the trade search once the deadline has passed |
| | `store.py`, `backtest.py` | One snapshot per day of projections, odds and trade ideas (`snap_*`) → accuracy by weeks-ahead, weekly win-probability calibration, title odds over time |
| | `offers.py`, `digest.py` | Offer log with outcomes → predicted vs observed acceptance; post-refresh digest |

Ranking: **EV = Δ P(title) × P(accept) × P(no veto)**.

## Honest limitations

- The acceptance and veto models are hand-set logistic curves (`ACC_*`, `VETO_*` in `.env`); there are too few trades
  in league history to fit them. Treat P(accept) as a ranking signal, not a calibrated probability. The offer log
  only suggests a re-centred `ACC_BIAS` once 8 offers are resolved.
- The horizon backtest and the simulation calibration start empty and fill in as daily snapshots accumulate.
- Playoff rounds in progress rely on ESPN returning playoff-week scores in the team schedule; that path is only
  tested on synthetic data. A finished round with no stored scores is simulated as a coin flip (and logged).
- Playoff odds are more confident than ESPN's because they use roster strength, not just record. The knobs
  (`PLAYER_SHOCK_SD`, `TEAM_SHOCK_SD`, `INJURY_RATE`) are documented in `config.py`.
- Sleeper's projection endpoint and ESPN's raw views are unofficial; each source fails independently.
- Ensemble weights (ESPN 0.42 / Sleeper 0.51 / usage 0.07) are the squared-error fit on weeks 2-9. The optimum is
  flat, so they beat other sensible blends by less than the noise; rerun `aniston backtest` as weeks accrue.
