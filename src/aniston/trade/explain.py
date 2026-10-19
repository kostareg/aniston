"""Plain-language reasons a trade works: why it helps you, why it looks good to them."""
import numpy as np


def _player_notes(ctx, i: int) -> list[str]:
    p = ctx.players.row(i, named=True)
    pv = ctx.pv.row(i, named=True)
    notes = []
    espn, ens = ctx.proj.espn[i].sum(), ctx.proj.ens[i].sum()
    if espn > 20 and abs(ens - espn) / espn >= 0.12:
        notes.append(f"ESPN {'over' if espn > ens else 'under'}rates ROS ({espn:.0f} vs consensus {ens:.0f})")
    fp, xfp = p.get("fp_pg"), p.get("xfp_pg")
    if fp is not None and xfp is not None and (p.get("games") or 0) >= 3 and abs(fp - xfp) >= 2.5:
        notes.append(f"{'running hot' if fp > xfp else 'unlucky'}: {fp:.1f} pts/g on {xfp:.1f} expected")
    if pv["last3"] is not None and pv["ppg"] and pv["last3"] - pv["ppg"] >= 4:
        notes.append(f"recent spike ({pv['last3']:.1f} last 3 vs {pv['ppg']:.1f} season)")
    if p["injury_status"] not in ("ACTIVE", "NORMAL", None):
        notes.append(f"status {p['injury_status']}")
    if pv["league_pick"] is not None and pv["league_pick"] <= 36:
        notes.append(f"round-{int((pv['league_pick'] - 1) // 12) + 1} draft cost")
    return notes


def rationale(ctx, t) -> str:
    nm = ctx.names
    parts = [f"You: +{t.my_gain:.0f} consensus ROS lineup pts"]
    if t.d_title is not None:
        parts[0] += f", title odds {t.d_title * 100:+.1f} pts"
    if t.my_moves.get("add"):
        parts.append("open spot -> pick up " + ", ".join(nm[i] for i in t.my_moves["add"]))
    if t.my_moves.get("drop"):
        parts.append("you drop " + ", ".join(nm[i] for i in t.my_moves["drop"]))
    parts.append(f"They see +{t.their_seen_gain:.0f} ESPN-projected lineup pts and {t.their_pvor_gain:+.0f} value; "
                 f"consensus says {t.their_true_gain:+.0f}")
    their_need = _need(ctx, t.partner, t.give)
    if their_need:
        parts.append(their_need)
    for i in t.get + t.give:
        notes = _player_notes(ctx, i)
        if notes:
            parts.append(f"{nm[i]}: " + "; ".join(notes))
    return " | ".join(parts)


def _need(ctx, partner: int, incoming) -> str | None:
    """Does what they receive fill their thinnest starting position (by ESPN projections)?"""
    roster = list(ctx.base[partner])
    spec = ctx.sim.spec
    weakest, gap = None, 0.0
    for pos_name, n in spec.dedicated.items():
        if pos_name in ("K", "D/ST"):
            continue
        mine = sorted((ctx.proj.espn[i].sum() for i in roster if ctx.pos[i] == pos_name), reverse=True)
        league = []
        for r in ctx.base.values():
            vals = sorted((ctx.proj.espn[i].sum() for i in r if ctx.pos[i] == pos_name), reverse=True)
            league.append(np.sum(vals[:n]))
        g = np.mean(league) - np.sum(mine[:n])
        if g > gap:
            weakest, gap = pos_name, g
    if weakest and any(ctx.pos[i] == weakest for i in incoming):
        return f"fills their weakest spot ({weakest}, {gap:.0f} pts below league avg)"
    return None
