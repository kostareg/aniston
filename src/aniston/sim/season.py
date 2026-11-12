"""Monte Carlo of the rest of the season: regular season -> seeding -> playoff bracket.

Randomness, all drawn once and shared across every roster configuration (common random numbers), so
the difference between "with trade" and "without trade" is not swamped by sampling noise:
  * a persistent per-player shock to their rest-of-season mean (projection error),
  * a per-player injury spell (onset week + duration) removing them from the lineup,
  * per team-week score noise scaled by the starting lineup's variance.
Each team starts its optimal lineup every week from whoever is available.

Weeks already played come from the stored results and a week under way from ESPN's live team projections, in
the regular season and in the playoff bracket alike; rosters only change the weeks in `proj.weeks`.
"""
import logging
import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from aniston import store
from aniston.config import settings
from aniston.model.lineup import LineupSpec
from aniston.model.projections import Projections
from aniston.model.variance import position_cv

log = logging.getLogger(__name__)

MEAN_INJURY_WEEKS = 3.0
CURRENT_WEEK_TEAM_CV = 0.22  # full-week team score SD / mean: the simulated future weeks and this season's actual spread


def remaining_sd(projected: float, score: float) -> float:
    """SD of the points a team has still to score this week. Independent player outcomes make it grow like the
    square root of the points left: a full week gives the team CV, one 12-point player left about his own CV."""
    return CURRENT_WEEK_TEAM_CV * math.sqrt(max(projected - score, 0.0) * max(projected, 0.0))


def onset_hazard(rate: float, n_weeks: int, mean_weeks: float = MEAN_INJURY_WEEKS) -> float:
    """Weekly injury-onset probability at which one spell (geometric onset, geometric duration with mean
    `mean_weeks`) costs a currently healthy player `rate` of the next `n_weeks`. Everyone starts healthy, so
    the naive rate / mean_weeks undershoots: about 2/3 of `rate` over six weeks."""
    if rate <= 0 or n_weeks <= 0:
        return 0.0
    stay = 1.0 - 1.0 / mean_weeks  # P(a spell lasts another week)
    t = np.arange(n_weeks)

    def share(q: float) -> float:
        # P(hurt in week t) = sum over onset k <= t of P(onset = k) * P(duration > t - k)
        k = np.arange(n_weeks)
        p = np.where(k[None, :] <= t[:, None], q * (1 - q) ** k[None, :] * stay ** (t[:, None] - k[None, :]), 0.0)
        return float(p.sum(1).mean())

    lo, hi = 0.0, 1.0
    if share(hi) <= rate:  # a single spell can't miss that much; injure everyone at once
        return 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if share(mid) < rate else (lo, mid)
    return (lo + hi) / 2


@dataclass
class SimResult:
    team_ids: list[int]
    p_playoffs: np.ndarray
    p_bye: np.ndarray
    p_title: np.ndarray
    exp_wins: np.ndarray
    seed_dist: np.ndarray  # (T, playoff_teams + 1): last column = missed playoffs
    title_by_sim: np.ndarray  # (S,) winning team index, for paired deltas
    p_win_next: np.ndarray  # (T,) P(win the next regular-season game); NaN once the regular season is over

    def frame(self) -> pl.DataFrame:
        cols = {f"seed_{i + 1}": self.seed_dist[:, i] for i in range(self.seed_dist.shape[1] - 1)}
        return pl.DataFrame({"team_id": self.team_ids, "p_playoffs": self.p_playoffs, "p_bye": self.p_bye,
                             "p_title": self.p_title, "exp_wins": self.exp_wins, "p_win_next": self.p_win_next,
                             **cols})


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
        hazard = {p: onset_hazard(s.injury_rate.get(p, 0.0), self.W) for p in np.unique(self.pos)}
        q = np.clip(np.array([hazard[p] for p in self.pos]), 1e-9, 1.0)
        onset = rng.geometric(q, (self.S, P)) - 1  # weeks from now until an injury starts
        dur = rng.geometric(1.0 / MEAN_INJURY_WEEKS, (self.S, P))
        w = np.arange(self.W)[None, :, None]
        hurt = (w >= onset[:, None, :]) & (w < (onset + dur)[:, None, :])
        self.means = (proj.ens.T[None, :, :] * (1.0 + shock[:, None, :]) * ~hurt).astype(np.float32)  # (S, W, P)
        self.z = rng.standard_normal((self.S, self.W, self.T)).astype(np.float32)
        self.z_cur = rng.standard_normal((self.S, self.T)).astype(np.float32)
        self.team_shock = (1.0 + rng.normal(0.0, s.team_shock_sd, (self.S, 1, self.T))).astype(np.float32)

        self.z_lost = rng.standard_normal((self.S, self.T)).astype(np.float32)  # playoff rounds with no stored score

        self._load_schedule()
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def _load_schedule(self):
        sched = store.read("schedule")
        last = self.meta["last_week"]
        if self.cur > self.reg_last:  # playoffs: seed from the regular season only, whatever ESPN's totals include
            reg = sched.filter(pl.col("week") <= self.reg_last)
            for t, i in self.tix.items():
                mine = reg.filter(pl.col("team_id") == t)
                self.base_wins[i] = (mine["outcome"] == "W").sum() + 0.5 * (mine["outcome"] == "T").sum()
                self.base_pf[i] = mine["score"].sum()
        self.reg_idx = [j for j, w in enumerate(self.weeks) if w <= self.reg_last]
        self.future_games = []  # (week_index, team_index_a, team_index_b), each game once
        for wk in (w for w in self.weeks if w <= self.reg_last):
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
        cur_live = self.cur not in self.weeks and self.cur <= last  # under way: scores come from ESPN, not rosters
        if cur_live and live.height:
            seen = set()
            for t, o, score, proj in live.select("team_id", "opp_id", "live_score", "live_projected").iter_rows():
                i = self.tix[t]
                self.cur_mean[i] = proj
                self.cur_sd[i] = remaining_sd(proj, score)
                key = tuple(sorted((t, o)))
                if self.cur <= self.reg_last and key not in seen:
                    seen.add(key)
                    self.cur_games.append((self.tix[t], self.tix[o]))
        self.next_week = self.cur if self.cur_games else min((self.weeks[w] for w, _, _ in self.future_games), default=None)
        # Playoff weeks no roster can change any more: week -> (mean, sd) per team. Finished weeks are exact.
        self.playoff_fixed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for wk in range(self.reg_last + 1, last + 1):
            if wk in self.weeks:
                continue
            if wk == self.cur:
                self.playoff_fixed[wk] = (self.cur_mean, self.cur_sd)
                continue
            done = sched.filter(pl.col("week") == wk)
            mean, sd = np.zeros(self.T), np.zeros(self.T)
            if done.height:
                for t, score in done.select("team_id", "score").iter_rows():
                    mean[self.tix[t]] = score
            else:  # ESPN gave no scores for a finished round: its winners are unknown, so they come out random
                log.warning("no stored scores for playoff week %s; simulating that round as a coin flip", wk)
                sd += 1.0
            self.playoff_fixed[wk] = (mean, sd)

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
        if self.cur <= self.reg_last:
            pf += cur
        p_win_next = np.full(T, np.nan)
        for a, b in self.cur_games:
            p_win_next[a], p_win_next[b] = (cur[:, a] > cur[:, b]).mean(), (cur[:, b] > cur[:, a]).mean()
        for w, a, b in self.future_games:
            sa, sb = scores[:, w, a], scores[:, w, b]
            wins[:, a] += sa > sb
            wins[:, b] += sb > sa
            if self.weeks[w] == self.next_week:
                p_win_next[a], p_win_next[b] = (sa > sb).mean(), (sb > sa).mean()
        pf += scores[:, self.reg_idx, :].sum(1)

        order = np.argsort(-(wins * 1e5 + pf), axis=1)  # seed tiebreak: total points
        seeds = order[:, : self.n_playoff]
        playoff = []  # (S, T) scores for each playoff week, in order
        for wk in range(self.reg_last + 1, self.meta["last_week"] + 1):
            if wk in self.playoff_fixed:
                mean, sd = self.playoff_fixed[wk]
                playoff.append(mean[None, :] + (self.z_cur * sd[None, :] if wk == self.cur else
                                                self.z_lost * sd[None, :]))
            else:
                playoff.append(scores[:, self.weeks.index(wk), :])
        champ = self._bracket(seeds, np.stack(playoff, axis=1))

        seed_dist = np.zeros((T, self.n_playoff + 1))
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(T)[None, :].repeat(S, 0), axis=1)
        for k in range(self.n_playoff):
            seed_dist[:, k] = (rank == k).mean(0)
        seed_dist[:, -1] = (rank >= self.n_playoff).mean(0)
        n_byes = {6: 2}.get(self.n_playoff, 0)
        return SimResult(
            team_ids=self.team_ids,
            p_playoffs=1 - seed_dist[:, -1],
            p_bye=seed_dist[:, :n_byes].sum(1),
            p_title=np.bincount(champ, minlength=T) / S,
            exp_wins=wins.mean(0),
            seed_dist=seed_dist,
            title_by_sim=champ,
            p_win_next=p_win_next,
        )

    def _bracket(self, seeds: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Single-week rounds over the playoff weeks; fixed bracket, no reseeding (8 teams: 1/8 vs 4/5, 2/7 vs 3/6;
        6 teams: 1 vs 4/5 winner, 2 vs 3/6 winner). scores: (S, playoff weeks, T)."""
        S = seeds.shape[0]
        rows = np.arange(S)
        pw = list(range(self.meta["last_week"] - self.reg_last))

        def game(a, b, w):
            return np.where(scores[rows, w, a] >= scores[rows, w, b], a, b)

        if self.n_playoff == 8 and len(pw) == 3:
            q18 = game(seeds[:, 0], seeds[:, 7], pw[0])
            q45 = game(seeds[:, 3], seeds[:, 4], pw[0])
            q27 = game(seeds[:, 1], seeds[:, 6], pw[0])
            q36 = game(seeds[:, 2], seeds[:, 5], pw[0])
            f1 = game(q18, q45, pw[1])
            f2 = game(q27, q36, pw[1])
            return game(f1, f2, pw[2])
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
