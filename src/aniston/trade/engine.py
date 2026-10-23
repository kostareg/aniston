"""Trade finder: enumerate packages with every other team, score them with fast proxies, then run the full
season simulation on the most promising ones.

  my true gain        = consensus-projection ROS lineup points after vs before (after legal drops/pickups)
  their seen gain     = ESPN-projection ROS lineup points for *their* roster after vs before
  their value gain    = perceived value over replacement they receive minus what they give
  P(accept), P(no veto) from those perceived quantities
  final ranking       = Δ P(title) for me (paired simulation) x P(accept) x P(no veto)
"""
import multiprocessing
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import polars as pl

from aniston import store
from aniston.config import settings
from aniston.model.projections import Projections, build
from aniston.sim.season import Simulator, SimResult, current_rosters
from aniston.trade.acceptance import p_accept, p_no_veto
from aniston.trade.perceived import perceived_values
from aniston.trade.roster import RosterValuer


@dataclass
class Trade:
    partner: int
    give: tuple[int, ...]  # universe indices I send
    get: tuple[int, ...]  # universe indices I receive
    my_roster: tuple[int, ...] = ()
    their_roster: tuple[int, ...] = ()
    my_moves: dict = field(default_factory=dict)
    their_moves: dict = field(default_factory=dict)
    my_gain: float = 0.0
    their_true_gain: float = 0.0
    their_seen_gain: float = 0.0
    their_pvor_gain: float = 0.0
    p_accept: float = 0.0
    p_no_veto: float = 0.0
    proxy: float = 0.0
    d_title: float | None = None
    d_title_se: float | None = None
    d_playoffs: float | None = None
    d_wins: float | None = None
    partner_d_title: float | None = None
    ev: float | None = None


class TradeContext:
    def __init__(self, proj: Projections | None = None, n_sims: int | None = None):
        s = settings()
        self.meta = store.get_json("meta")
        self.me = self.meta["my_team_id"]
        self.proj = proj or build()
        self.sim = Simulator(self.proj, n_sims=n_sims)
        self.players = self.proj.players
        self.pos = self.players["pos"].to_numpy()
        self.names = self.players["name"].to_list()
        injury = self.players["injury_status"].to_numpy()
        active_cap = sum(self.meta["slots"].values()) + self.meta["bench"]
        args = dict(pos=self.pos, injury=injury, spec=self.sim.spec, weeks=self.proj.weeks,
                    reg_last=self.meta["reg_season_weeks"], active_cap=active_cap, ir_slots=self.meta["ir"])
        self.true = RosterValuer(self.proj.ens, **args)
        self.seen = RosterValuer(self.proj.espn, **args)

        pv = perceived_values(self.proj)
        self.pv = pv
        self.pvor = pv["pvor"].to_numpy()
        # Other managers cut whoever *looks* least valuable (an injured star keeps his name value).
        self.seen.keep_score = pv["pv"].to_numpy()
        self.free_agents = [i for i, t in enumerate(self.players["fantasy_team_id"].to_list()) if t == 0]
        self.tradeable = self.players["tradeable"].to_numpy()

        self.base = self._legal_baseline(current_rosters(self.proj))
        self.base_result = self.sim.run(self.base)
        traded = store.read("trades")["team_id"].unique().to_list() if store.has("trades") else []
        self.has_traded = set(traded)
        self.teams = dict(store.read("teams").select("team_id", "name").iter_rows())

    def _legal_baseline(self, rosters: dict[int, list[int]]) -> dict[int, tuple[int, ...]]:
        """Fill open roster spots (and trim overfull ones) before any trade, so trades get no credit for
        a pickup that was available anyway. Each team claims distinct free agents."""
        taken: set[int] = set()
        out = {}
        for t in sorted(rosters, key=lambda t: t != self.me):  # me first: I pick by consensus value
            valuer = self.true if t == self.me else self.seen
            out[t], _, added = valuer.legalize(tuple(rosters[t]), [f for f in self.free_agents if f not in taken])
            taken.update(added)
        self.free_agents = [f for f in self.free_agents if f not in taken]
        return out

    # ------------------------------------------------------------------ proxies
    def build_trade(self, partner: int, give: tuple[int, ...], get: tuple[int, ...],
                    min_my_gain: float | None = None) -> Trade | None:
        """Score a trade with the fast proxies. Returns None early when my gain is below `min_my_gain`."""
        t = Trade(partner=partner, give=give, get=get)
        mine = tuple(i for i in self.base[self.me] if i not in give) + get
        theirs = tuple(i for i in self.base[partner] if i not in get) + give
        t.my_roster, d1, a1 = self.true.legalize(mine, self.free_agents, protected=set(get))
        t.my_gain = self.true.value(t.my_roster) - self.true.value(self.base[self.me])
        if min_my_gain is not None and t.my_gain <= min_my_gain:
            return None
        # The partner manages their roster by what they see (ESPN), and can't take my pickups.
        t.their_roster, d2, a2 = self.seen.legalize(theirs, [f for f in self.free_agents if f not in a1],
                                                    protected=set(give))
        t.my_moves, t.their_moves = {"drop": d1, "add": a1}, {"drop": d2, "add": a2}
        t.their_true_gain = self.true.value(t.their_roster) - self.true.value(self.base[partner])
        t.their_seen_gain = self.seen.value(t.their_roster) - self.seen.value(self.base[partner])
        t.their_pvor_gain = float(self.pvor[list(give)].sum() - self.pvor[list(get)].sum())
        t.p_accept = p_accept(t.their_seen_gain, t.their_pvor_gain, partner in self.has_traded)
        t.p_no_veto = p_no_veto(-t.their_pvor_gain)
        t.proxy = t.my_gain * t.p_accept * t.p_no_veto
        return t

    def _pool(self, team: int, size: int | None, exclude: set[int]) -> list[int]:
        r = [i for i in self.base[team] if self.tradeable[i] and self.pos[i] not in ("K", "D/ST") and i not in exclude]
        r.sort(key=lambda i: -self.true.ros[i])
        return r[:size] if size else r

    def candidates(self, partners: list[int] | None = None, max_players: int = 2,
                   untouchable: set[int] = frozenset(), must_include: set[int] = frozenset(),
                   workers: int = 1) -> list[Trade]:
        """`workers` > 1 scores partners in forked processes; only use it outside Streamlit (forking a
        threaded server is unsafe)."""
        partners = partners or [t for t in self.base if t != self.me]
        args = [(p, max_players, untouchable, must_include) for p in partners]
        if workers > 1 and len(partners) > 1:
            global _FORK_CTX
            _FORK_CTX = self
            try:
                with multiprocessing.get_context("fork").Pool(min(workers, len(partners))) as pool:
                    per_partner = pool.starmap(_partner_candidates, args, chunksize=1)
            finally:
                _FORK_CTX = None
        else:
            per_partner = [self._partner_candidates(*a) for a in args]
        out = [t for ts in per_partner for t in ts]
        out.sort(key=lambda t: -t.proxy)
        return out

    def _partner_candidates(self, p: int, max_players: int, untouchable: set[int],
                            must_include: set[int]) -> list[Trade]:
        s = settings()
        my_all = self._pool(self.me, None, untouchable)
        my_top = my_all[: s.trade_pool_size]
        their_all = self._pool(p, None, set())
        their_top = their_all[: s.trade_pool_size]
        out = []
        shapes = [(1, 1)] + ([(2, 1), (1, 2), (2, 2)] if max_players >= 2 else [])
        for ng, nr in shapes:
            gives = combinations(my_all if ng == 1 else my_top, ng)
            for give in gives:
                if must_include and not must_include & set(give):
                    continue
                for get in combinations(their_all if nr == 1 else their_top, nr):
                    t = self.build_trade(p, give, get, min_my_gain=0.0)
                    if t is not None and t.p_accept >= s.trade_min_accept:
                        out.append(t)
        return out

    # ------------------------------------------------------------------ simulation
    def evaluate(self, t: Trade) -> Trade:
        rosters = dict(self.base)
        rosters[self.me], rosters[t.partner] = t.my_roster, t.their_roster
        res: SimResult = self.sim.run(rosters)
        b = self.base_result
        mi, pi = self.sim.tix[self.me], self.sim.tix[t.partner]
        paired = (res.title_by_sim == mi).astype(float) - (b.title_by_sim == mi).astype(float)
        t.d_title = float(paired.mean())
        t.d_title_se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
        t.d_playoffs = float(res.p_playoffs[mi] - b.p_playoffs[mi])
        t.d_wins = float(res.exp_wins[mi] - b.exp_wins[mi])
        t.partner_d_title = float(res.p_title[pi] - b.p_title[pi])
        t.ev = t.d_title * t.p_accept * t.p_no_veto
        return t

    def find(self, top_k: int | None = None, per_partner: int = 3, **kw) -> list[Trade]:
        top_k = top_k or settings().trade_sim_top_k
        cands = self.candidates(**kw)
        chosen, per = [], {}
        for t in cands:  # keep the shortlist diverse across partners
            if per.get(t.partner, 0) < per_partner * 3:
                chosen.append(t)
                per[t.partner] = per.get(t.partner, 0) + 1
            if len(chosen) >= top_k:
                break
        evaluated = [self.evaluate(t) for t in chosen]
        evaluated.sort(key=lambda t: -(t.ev or 0))
        return evaluated

    def frame(self, trades: list[Trade]) -> pl.DataFrame:
        from aniston.trade.explain import rationale

        nm = lambda ids: " + ".join(self.names[i] for i in ids)  # noqa: E731
        return pl.DataFrame([{
            "partner_id": t.partner, "partner": self.teams[t.partner], "give": nm(t.give), "get": nm(t.get),
            "you_drop": nm(t.my_moves.get("drop", [])), "you_add": nm(t.my_moves.get("add", [])),
            "they_drop": nm(t.their_moves.get("drop", [])), "they_add": nm(t.their_moves.get("add", [])),
            "d_title": t.d_title, "d_title_se": t.d_title_se, "d_playoffs": t.d_playoffs, "d_wins": t.d_wins,
            "my_ros_gain": t.my_gain, "their_seen_gain": t.their_seen_gain, "their_true_gain": t.their_true_gain,
            "their_value_gain": t.their_pvor_gain, "partner_d_title": t.partner_d_title,
            "p_accept": t.p_accept, "p_no_veto": t.p_no_veto, "ev": t.ev, "why": rationale(self, t),
            "give_ids": [int(self.players["espn_id"][i]) for i in t.give],
            "get_ids": [int(self.players["espn_id"][i]) for i in t.get],
        } for t in trades])


_FORK_CTX: TradeContext | None = None  # inherited by forked workers, so the context is never pickled


def _partner_candidates(*args) -> list[Trade]:
    return _FORK_CTX._partner_candidates(*args)
