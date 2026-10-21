from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    espn_s2: str
    swid: str
    league_id: int
    season: int = 2026
    my_team_id: int | None = None  # derived from SWID when unset
    db_path: Path = Path("/var/lib/aniston/aniston.duckdb")

    # Projection ensemble weights (renormalized over available sources per player-week). Backtest on weeks
    # 2-5 of 2026 fit ESPN 0.36 / Sleeper 0.64 / usage 0.0; usage keeps a small share for its ROS signal.
    w_espn: float = 0.38
    w_sleeper: float = 0.55
    w_usage: float = 0.07

    # Simulation.
    n_sims: int = 5000
    seed: int = 7
    player_shock_sd: float = 0.20  # persistent rest-of-season error in a player's true mean
    team_shock_sd: float = 0.04  # persistent team-level drift: waivers, lineup mistakes, other managers' moves
    # Share of future games missed to new injuries (on top of what projections already know).
    injury_rate: dict[str, float] = {"QB": 0.06, "RB": 0.11, "WR": 0.09, "TE": 0.08, "K": 0.0, "D/ST": 0.0}

    # Perceived value: what a league-mate sees in the ESPN app.
    pv_w_espn_ros: float = 0.45
    pv_w_season_ppg: float = 0.25
    pv_w_last3_ppg: float = 0.15
    pv_w_adp: float = 0.15

    # Acceptance model: logistic over the partner's perceived gains.
    acc_bias: float = -0.4
    acc_w_lineup: float = 0.06  # per perceived ROS starting-lineup point gained
    acc_w_value: float = 0.025  # per perceived value-over-replacement point gained
    acc_active_bonus: float = 0.3  # managers who have already traded this season are likelier to deal
    # League veto (ESPN vote): trades that look lopsided in your favour risk a veto.
    veto_bias: float = 3.0
    veto_w: float = 0.05  # per perceived value point you appear to win by

    # Trade search.
    trade_pool_size: int = 10  # each side's top-N players (by consensus ROS) considered for 2-player packages
    trade_sim_top_k: int = 60  # candidates that get the full season simulation
    trade_min_accept: float = 0.10


@lru_cache
def settings() -> Settings:
    return Settings()
