"""Monte Carlo of the rest of the season: regular season -> seeding -> playoff bracket.

Randomness, all drawn once and shared across every roster configuration (common random numbers), so
the difference between "with trade" and "without trade" is not swamped by sampling noise:
  * a persistent per-player shock to their rest-of-season mean (projection error),
  * a per-player injury spell (onset week + duration) removing them from the lineup,
  * per team-week score noise scaled by the starting lineup's variance.
Each team starts its optimal lineup every week from whoever is available.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from aniston import store
from aniston.config import settings
from aniston.model.lineup import LineupSpec
from aniston.model.projections import Projections
from aniston.model.variance import position_cv

MEAN_INJURY_WEEKS = 3.0
CURRENT_WEEK_REMAINING_CV = 0.35


@dataclass
class SimResult:
    team_ids: list[int]
    p_playoffs: np.ndarray
    p_bye: np.ndarray
    p_title: np.ndarray
    exp_wins: np.ndarray
    seed_dist: np.ndarray  # (T, playoff_teams + 1): last column = missed playoffs
    title_by_sim: np.ndarray  # (S,) winning team index, for paired deltas

    def frame(self) -> pl.DataFrame:
        cols = {f"seed_{i + 1}": self.seed_dist[:, i] for i in range(self.seed_dist.shape[1] - 1)}
        return pl.DataFrame({"team_id": self.team_ids, "p_playoffs": self.p_playoffs, "p_bye": self.p_bye,
                             "p_title": self.p_title, "exp_wins": self.exp_wins, **cols})


class Simulator:
    def __init__(self, proj: Projections, n_sims: int | None = None, seed: int | None = None):
        s = settings()
        self.meta = store.get_json("meta")
        self.proj = proj
        self.S = n_sims or s.n_sims
        rng = np.random.default_rng(seed if seed is not None else s.seed)

        teams = store.read("teams").sort("team_id")
        self.team_ids = teams["team_id"].to_list()
        self.tix = {t: i for i, t in enumerate(self.team_ids)}
        self.T = len(self.team_ids)
        self.base_wins = teams["wins"].to_numpy().astype(float) + 0.5 * teams["ties"].to_numpy()
        self.base_pf = teams["points_for"].to_numpy().astype(float)

        self.weeks = proj.weeks
        self.W = len(self.weeks)
        self.cur = self.meta["current_week"]
        self.reg_last = self.meta["reg_season_weeks"]
        self.n_playoff = self.meta["playoff_teams"]
        self.pos = proj.players["pos"].to_numpy()
        self.spec = LineupSpec(self.meta["slots"], position_cv(s.season))

        P = proj.players.height
        shock = np.clip(rng.normal(0.0, s.player_shock_sd, (self.S, P)), -0.8, 1.0)
        rate = np.array([s.injury_rate.get(p, 0.0) for p in self.pos])
        q = np.clip(rate / MEAN_INJURY_WEEKS, 1e-9, 1.0)
        onset = rng.geometric(q, (self.S, P)) - 1  # weeks from now until an injury starts
        dur = rng.geometric(1.0 / MEAN_INJURY_WEEKS, (self.S, P))
        w = np.arange(self.W)[None, :, None]
        hurt = (w >= onset[:, None, :]) & (w < (onset + dur)[:, None, :])
        self.means = (proj.ens.T[None, :, :] * (1.0 + shock[:, None, :]) * ~hurt).astype(np.float32)  # (S, W, P)
        self.z = rng.standard_normal((self.S, self.W, self.T)).astype(np.float32)
        self.z_cur = rng.standard_normal((self.S, self.T)).astype(np.float32)
        self.team_shock = (1.0 + rng.normal(0.0, s.team_shock_sd, (self.S, 1, self.T))).astype(np.float32)

        self._load_schedule()
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _load_schedule(self):
        sched = store.read("schedule")
        self.future_games = []  # (week_index, team_index_a, team_index_b), each game once
        for wk in range(self.cur + 1, self.reg_last + 1):
            seen = set()
            for t, o in sched.filter(pl.col("week") == wk).select("team_id", "opp_id").iter_rows():
                key = tuple(sorted((t, o)))
                if key not in seen:
                    seen.add(key)
                    self.future_games.append((self.weeks.index(wk), self.tix[t], self.tix[o]))
        live = store.read("live")
        self.cur_games = []
        self.cur_mean = np.zeros(self.T)
        self.cur_sd = np.zeros(self.T)
        if self.cur <= self.reg_last and live.height:
            seen = set()
            for t, o, score, proj in live.select("team_id", "opp_id", "live_score", "live_projected").iter_rows():
                i = self.tix[t]
                self.cur_mean[i] = proj
                self.cur_sd[i] = CURRENT_WEEK_REMAINING_CV * max(proj - score, 0.0)
                key = tuple(sorted((t, o)))
                if key not in seen:
                    seen.add(key)
                    self.cur_games.append((self.tix[t], self.tix[o]))

    def team_scores(self, roster_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(S, W) lineup mean and score SD for one roster (cached by roster contents)."""
        key = tuple(sorted(roster_idx))
        if key not in self._cache:
            r = np.array(key, dtype=int)
            total, var = self.spec.evaluate(self.means[:, :, r], self.pos[r], with_var=True)
            self._cache[key] = (total.astype(np.float32), np.sqrt(var).astype(np.float32))
        return self._cache[key]

    def run(self, rosters: dict[int, list[int]]) -> SimResult:
        S, T = self.S, self.T
        scores = np.empty((S, self.W, T), dtype=np.float32)
        for t, r in rosters.items():
            mean, sd = self.team_scores(r)
            i = self.tix[t]
            scores[:, :, i] = mean * self.team_shock[:, :, i] + self.z[:, :, i] * sd

        wins = np.tile(self.base_wins, (S, 1))
        pf = np.tile(self.base_pf, (S, 1))
        cur = self.cur_mean[None, :] + self.z_cur * self.cur_sd[None, :]
        for a, b in self.cur_games:
            wins[:, a] += cur[:, a] > cur[:, b]
            wins[:, b] += cur[:, b] > cur[:, a]
        pf += cur
        for w, a, b in self.future_games:
            sa, sb = scores[:, w, a], scores[:, w, b]
            wins[:, a] += sa > sb
            wins[:, b] += sb > sa
        reg_idx = [self.weeks.index(w) for w in range(self.cur + 1, self.reg_last + 1)]
        pf += scores[:, reg_idx, :].sum(1)

        order = np.argsort(-(wins * 1e5 + pf), axis=1)  # seed tiebreak: total points
        seeds = order[:, : self.n_playoff]
        champ = self._bracket(seeds, scores)

        seed_dist = np.zeros((T, self.n_playoff + 1))
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(T)[None, :].repeat(S, 0), axis=1)
        for k in range(self.n_playoff):
            seed_dist[:, k] = (rank == k).mean(0)
        seed_dist[:, -1] = (rank >= self.n_playoff).mean(0)
        n_byes = 2 if self.n_playoff == 6 else 0
        return SimResult(
            team_ids=self.team_ids,
            p_playoffs=1 - seed_dist[:, -1],
            p_bye=seed_dist[:, :n_byes].sum(1),
            p_title=np.bincount(champ, minlength=T) / S,
            exp_wins=wins.mean(0),
            seed_dist=seed_dist,
            title_by_sim=champ,
        )

    def _bracket(self, seeds: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Single-week rounds over the playoff weeks; fixed bracket (1 vs 4/5 winner, 2 vs 3/6 winner)."""
        S = seeds.shape[0]
        rows = np.arange(S)
        pw = [self.weeks.index(w) for w in range(self.reg_last + 1, self.meta["last_week"] + 1)]

        def game(a, b, w):
            return np.where(scores[rows, w, a] >= scores[rows, w, b], a, b)

        if self.n_playoff == 6 and len(pw) == 3:
            w45 = game(seeds[:, 3], seeds[:, 4], pw[0])
            w36 = game(seeds[:, 2], seeds[:, 5], pw[0])
            f1 = game(seeds[:, 0], w45, pw[1])
            f2 = game(seeds[:, 1], w36, pw[1])
            return game(f1, f2, pw[2])
        if self.n_playoff == 4 and len(pw) >= 2:
            f1 = game(seeds[:, 0], seeds[:, 3], pw[0])
            f2 = game(seeds[:, 1], seeds[:, 2], pw[0])
            return game(f1, f2, pw[1])
        raise NotImplementedError(f"bracket for {self.n_playoff} teams over {len(pw)} weeks")


def current_rosters(proj: Projections) -> dict[int, list[int]]:
    p = proj.players.with_row_index("i")
    out: dict[int, list[int]] = {}
    for i, t in p.filter(pl.col("fantasy_team_id") > 0).select("i", "fantasy_team_id").iter_rows():
        out.setdefault(t, []).append(i)
    return out
