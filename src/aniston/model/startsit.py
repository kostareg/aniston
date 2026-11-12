"""This week's lineup, and start/sit swaps that raise the chance of winning the matchup rather than the mean.

The points-optimal lineup is the right one against an even opponent. A clear underdog gains from volatility
(a boom-or-bust receiver over a steady back projected slightly higher) and a clear favourite from the opposite,
so each one-for-one swap is scored by the win probability it gives against this week's opponent.
"""
import math

import numpy as np
import polars as pl

from aniston import store
from aniston.model.variance import player_cv

MIN_WIN_GAIN = 0.002  # swaps worth showing: at least 0.2 points of win probability


def win_prob(mu_a: float, sd_a: float, mu_b: float, sd_b: float) -> float:
    sd = math.hypot(sd_a, sd_b)
    if sd == 0:  # every game is final: the result is decided
        return 1.0 if mu_a > mu_b else 0.0 if mu_a < mu_b else 0.5
    return 0.5 * (1 + math.erf((mu_a - mu_b) / (sd * math.sqrt(2))))


def _eligible(spec, slot: str, pos: str) -> bool:
    return pos == slot or any(name == slot and pos in elig for elig, name in spec.flex_names.items())


def lineup_advice(ctx) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Returns (my lineup for the current week, suggested swaps, matchup summary)."""
    spec, me = ctx.sim.spec, ctx.me
    cv = player_cv(ctx.players, spec.cv)
    proj = ctx.proj.ens_cur

    def team(t: int):
        r = np.array(ctx.base[t])
        slots = spec.slots(proj[r], ctx.pos[r])
        start = slots != ""
        return r, slots, float(proj[r][start].sum()), float(((cv[r] * proj[r]) ** 2)[start].sum())

    r, slots, mu, var = team(me)
    lineup = pl.DataFrame({
        "espn_id": ctx.players["espn_id"].to_numpy()[r], "name": [ctx.names[i] for i in r],
        "pos": ctx.pos[r].tolist(), "slot": slots.tolist(), "proj": proj[r], "sd": cv[r] * proj[r],
        "injury_status": ctx.players["injury_status"].to_numpy()[r].tolist(),
    })
    info = {"week": ctx.proj.cur_week, "my_mean": mu, "my_sd": math.sqrt(var)}

    opp = None
    if store.has("live"):
        row = store.read("live").filter(pl.col("team_id") == me)
        opp = row["opp_id"][0] if row.height else None
    swaps = []
    if opp in ctx.base:
        _, _, mu_o, var_o = team(opp)
        base = win_prob(mu, math.sqrt(var), mu_o, math.sqrt(var_o))
        info |= {"opp_id": int(opp), "opp_mean": mu_o, "opp_sd": math.sqrt(var_o), "win_prob": base}
        for a in np.where(slots != "")[0]:
            for b in np.where(slots == "")[0]:
                if proj[r[b]] <= 0 or not _eligible(spec, slots[a], ctx.pos[r[b]]):
                    continue
                d_mu = proj[r[b]] - proj[r[a]]
                new_var = var - (cv[r[a]] * proj[r[a]]) ** 2 + (cv[r[b]] * proj[r[b]]) ** 2
                p = win_prob(mu + d_mu, math.sqrt(new_var), mu_o, math.sqrt(var_o))
                if p - base >= MIN_WIN_GAIN:
                    swaps.append({"sit": ctx.names[r[a]], "start": ctx.names[r[b]], "slot": slots[a],
                                  "d_proj": float(d_mu), "d_sd": math.sqrt(new_var) - math.sqrt(var),
                                  "d_win": p - base})
    swaps.sort(key=lambda s: -s["d_win"])
    schema = {"sit": pl.Utf8, "start": pl.Utf8, "slot": pl.Utf8, "d_proj": pl.Float64, "d_sd": pl.Float64,
              "d_win": pl.Float64}
    return lineup, pl.DataFrame(swaps, schema=schema), info
