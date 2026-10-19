"""Roster valuation and legality: a roster's rest-of-season lineup points, drops when over the limit,
best free-agent pickup when a spot opens."""
from functools import lru_cache

import numpy as np

from aniston.model.lineup import LineupSpec

IR_ELIGIBLE = {"OUT", "INJURY_RESERVE", "SUSPENSION"}
PLAYOFF_WEEK_WEIGHT = 1.5  # title odds hinge on playoff weeks, so weight them up in quick proxies


class RosterValuer:
    """Values rosters (tuples of universe indices) against one projection matrix (P, W)."""

    def __init__(self, mat: np.ndarray, pos: np.ndarray, injury: np.ndarray, spec: LineupSpec,
                 weeks: list[int], reg_last: int, active_cap: int, ir_slots: int):
        self.mat, self.pos, self.spec = mat, pos, spec
        self.ir_ok = np.isin(injury, list(IR_ELIGIBLE))
        self.w = np.where(np.array(weeks) > reg_last, PLAYOFF_WEEK_WEIGHT, 1.0)
        self.active_cap, self.ir_slots = active_cap, ir_slots
        self.ros = mat.sum(1)
        self.keep_score = self.ros  # what decides who gets cut; managers may judge by perceived value instead
        self.min_count = {p: n for p, n in spec.dedicated.items()}
        self.value = lru_cache(maxsize=200_000)(self._value)

    def _value(self, roster: tuple[int, ...]) -> float:
        r = np.fromiter(roster, dtype=int)
        return float(self.spec.evaluate(self.mat[r].T, self.pos[r]) @ self.w)

    def cap(self, roster) -> int:
        return self.active_cap + min(self.ir_slots, int(self.ir_ok[list(roster)].sum()))

    def drop_candidate(self, roster: tuple[int, ...], protected: set[int] = frozenset()) -> int:
        """Least valuable droppable player: lowest keep score at a position held beyond its starting requirement."""
        counts: dict[str, int] = {}
        for i in roster:
            counts[self.pos[i]] = counts.get(self.pos[i], 0) + 1
        choices = [i for i in roster if i not in protected and counts[self.pos[i]] > self.min_count.get(self.pos[i], 0)]
        return min(choices or [i for i in roster if i not in protected], key=lambda i: self.keep_score[i])

    def shortlist(self, free_agents: list[int], per_pos: int = 3) -> list[int]:
        """Top free agents per position by this valuer's ROS points; deeper options never win a pickup."""
        out, seen = [], {}
        for fa in sorted(free_agents, key=lambda i: -self.ros[i]):
            if seen.get(self.pos[fa], 0) < per_pos:
                seen[self.pos[fa]] = seen.get(self.pos[fa], 0) + 1
                out.append(fa)
        return out

    def best_pickup(self, roster: tuple[int, ...], free_agents: list[int]) -> int | None:
        """Free agent adding the most lineup points; when none adds any (pure depth), the best by ROS points.
        Managers always fill an open spot, and depth still matters once injuries are simulated."""
        base = self.value(tuple(sorted(roster)))
        best, key = None, (-np.inf, -np.inf)
        for fa in self.shortlist(free_agents):
            k = (round(self.value(tuple(sorted(roster + (fa,)))) - base, 6), self.ros[fa])
            if k > key:
                best, key = fa, k
        return best

    def legalize(self, roster: tuple[int, ...], free_agents: list[int], protected: set[int] = frozenset(),
                 allow_pickup: bool = True) -> tuple[tuple[int, ...], list[int], list[int]]:
        """Drop down to the roster limit, or pick up free agents to fill open spots.
        Returns (roster, dropped, added)."""
        roster = tuple(sorted(roster))
        dropped, added = [], []
        while len(roster) > self.cap(roster):
            d = self.drop_candidate(roster, protected)
            dropped.append(d)
            roster = tuple(i for i in roster if i != d)
        while allow_pickup and len(roster) < self.cap(roster):
            fa = self.best_pickup(roster, [f for f in free_agents if f not in added])
            if fa is None:
                break
            added.append(fa)
            roster = tuple(sorted(roster + (fa,)))
        return roster, dropped, added
