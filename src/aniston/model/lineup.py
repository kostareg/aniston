"""Optimal lineup selection, vectorized over any leading axes (sims, weeks).

Dedicated slots take the best players at their position, then flex slots take the best leftovers from
their eligible positions. Greedy is optimal here because every flex slot's eligible set contains the
dedicated positions it competes with (RB/WR/TE, OP, ...).
"""
import numpy as np

FLEX_ELIGIBLE = {
    "RB/WR/TE": ("RB", "WR", "TE"),
    "RB/WR": ("RB", "WR"),
    "WR/TE": ("WR", "TE"),
    "OP": ("QB", "RB", "WR", "TE"),
}


class LineupSpec:
    def __init__(self, slots: dict[str, int], cv: dict[str, float] | None = None):
        self.dedicated = {p: n for p, n in slots.items() if p not in FLEX_ELIGIBLE}
        self.flex = [(FLEX_ELIGIBLE[p], n) for p, n in slots.items() if p in FLEX_ELIGIBLE]
        self.cv = cv or {}

    def evaluate(self, values: np.ndarray, pos: np.ndarray, with_var: bool = False):
        """values: (..., n) projected points for a roster; pos: (n,) position strings.

        Returns total starting-lineup points (...,) and, if requested, the sum over starters of
        (cv_pos * points)^2 — the weekly score variance under independent player outcomes.
        """
        total = np.zeros(values.shape[:-1])
        var = np.zeros(values.shape[:-1])
        leftovers: dict[str, np.ndarray] = {}
        for p, n in self.dedicated.items():
            v = np.sort(values[..., pos == p], axis=-1)[..., ::-1]
            take = v[..., :n]
            total += take.sum(-1)
            if with_var:
                var += ((self.cv.get(p, 0.5) * take) ** 2).sum(-1)
            leftovers[p] = v[..., n:]
        for eligible, n in self.flex:
            parts = [leftovers[p] for p in eligible if p in leftovers and leftovers[p].shape[-1]]
            if not parts:
                continue
            pool = np.concatenate(parts, axis=-1)
            cvs = np.concatenate([np.full(leftovers[p].shape[-1], self.cv.get(p, 0.5))
                                  for p in eligible if p in leftovers and leftovers[p].shape[-1]])
            order = np.argsort(-pool, axis=-1)[..., :n]
            take = np.take_along_axis(pool, order, axis=-1)
            take = np.where(np.isfinite(take), take, 0.0)  # fewer eligible players than flex slots
            total += take.sum(-1)
            if with_var:
                var += ((cvs[order] * take) ** 2).sum(-1)
            # Remove the chosen players from every eligible leftover pool for subsequent flex slots.
            chosen = np.zeros(pool.shape, dtype=bool)
            np.put_along_axis(chosen, order, True, axis=-1)
            start = 0
            for p in eligible:
                if p in leftovers and leftovers[p].shape[-1]:
                    k = leftovers[p].shape[-1]
                    leftovers[p] = np.where(chosen[..., start:start + k], -np.inf, leftovers[p])
                    start += k
        return (total, var) if with_var else total

    def starters(self, values: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Boolean mask (n,) of starters for a single 1-D roster projection."""
        n_players = values.shape[-1]
        mask = np.zeros(n_players, dtype=bool)
        idx_by_pos = {p: np.where(pos == p)[0] for p in set(pos)}
        left: dict[str, list[int]] = {}
        for p, n in self.dedicated.items():
            ids = sorted(idx_by_pos.get(p, []), key=lambda i: -values[i])
            mask[ids[:n]] = True
            left[p] = ids[n:]
        for eligible, n in self.flex:
            pool = sorted((i for p in eligible for i in left.get(p, [])), key=lambda i: -values[i])[:n]
            mask[pool] = True
            for p in eligible:
                left[p] = [i for i in left.get(p, []) if i not in pool]
        return mask
