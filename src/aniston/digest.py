"""Short message after each refresh: odds, what changed in the lineup and on the injury report, the top trade
and waiver ideas. Sent to NOTIFY_URL (an ntfy topic or a Discord webhook); printed when that is unset.

Only reads tables written by `analyze`. What it reported last time lives in the `digest_state` key, so the
next digest can say what is new.
"""
import httpx
import polars as pl

from aniston import store
from aniston.config import deadline_passed, settings

HEALTHY = ("ACTIVE", "NORMAL", None)


def build() -> tuple[str, str, dict]:
    """Returns (title, body, state to remember)."""
    meta = store.get_json("meta")
    prev = store.get_json("digest_state") or {}
    me = meta["my_team_id"]
    lines = []

    sim = store.read("sim_baseline").filter(pl.col("team_id") == me).row(0, named=True)
    odds = f"Title {sim['p_title']:.1%}, playoffs {sim['p_playoffs']:.0%}"
    if prev.get("p_title") is not None:
        odds += f" (title {(sim['p_title'] - prev['p_title']) * 100:+.1f} pts since last digest)"
    lines.append(odds)

    info = store.get_json("lineup_info") or {}
    if info.get("win_prob") is not None:
        opp = dict(store.read("teams").select("team_id", "name").iter_rows()).get(info["opp_id"], "?")
        lines.append(f"Week {info['week']} vs {opp}: {info['my_mean']:.0f} to {info['opp_mean']:.0f} projected, "
                     f"{info['win_prob']:.0%} to win")

    lineup = store.read("my_lineup")
    starters = sorted(lineup.filter(pl.col("slot") != "")["name"].to_list())
    if prev.get("starters") and prev.get("lineup_week") == info.get("week"):
        came_in = [n for n in starters if n not in prev["starters"]]
        went_out = [n for n in prev["starters"] if n not in starters]
        if came_in or went_out:
            lines.append(f"Lineup change: start {', '.join(came_in) or '–'}; sit {', '.join(went_out) or '–'}")
    else:
        lines.append("Optimal lineup: " + ", ".join(
            lineup.filter(pl.col("slot") != "").sort("proj", descending=True)["name"].to_list()))

    swaps = store.read("start_sit")
    if swaps.height:
        s = swaps.row(0, named=True)
        lines.append(f"Start/sit: {s['start']} over {s['sit']} adds {s['d_win'] * 100:.1f} pts of win probability "
                     f"({s['d_proj']:+.1f} projected)")

    injuries = {n: st for n, st in lineup.select("name", "injury_status").iter_rows() if st not in HEALTHY}
    old = prev.get("injuries") or {}
    for name in sorted(set(injuries) | set(old)):
        if injuries.get(name) != old.get(name):
            was = f"{old.get(name, 'ACTIVE')} -> " if prev else ""  # the first digest has nothing to compare with
            lines.append(f"Injury: {name} {was}{injuries.get(name, 'ACTIVE')}")

    top_trade = None
    if not deadline_passed(meta) and store.has("trade_ideas"):
        ideas = store.read("trade_ideas")
        if ideas.height and "ev" in ideas.columns:
            t = ideas.sort("ev", descending=True).row(0, named=True)
            top_trade = f"{t['partner']}: give {t['give']}, get {t['get']}"
            new = "New top trade" if top_trade != prev.get("top_trade") else "Top trade (unchanged)"
            lines.append(f"{new}: {top_trade} (title {t['d_title'] * 100:+.1f} pts, accept {t['p_accept']:.0%})")

    top_waiver = None
    waivers = store.read("waivers").filter(pl.col("ros_lineup_gain") > 0)
    if waivers.height:
        w = waivers.row(0, named=True)
        top_waiver = w["name"]
        if top_waiver != prev.get("top_waiver"):
            lines.append(f"Waiver: add {w['name']} ({w['pos']}), drop {w['drop'] or '–'}: "
                         f"{w['ros_lineup_gain']:+.0f} ROS lineup pts")

    state = {"p_title": sim["p_title"], "starters": starters, "lineup_week": info.get("week"),
             "injuries": injuries, "top_trade": top_trade, "top_waiver": top_waiver}
    return f"Aniston: week {meta['current_week']}", "\n".join(lines), state


def send(title: str, body: str) -> str:
    url = settings().notify_url
    if not url:
        return "not sent: NOTIFY_URL is unset"
    if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
        r = httpx.post(url, json={"content": f"**{title}**\n{body}"[:2000]}, timeout=20)
    else:  # ntfy: the message is the body, the title a header
        r = httpx.post(url, content=body.encode(), headers={"Title": title}, timeout=20)
    r.raise_for_status()
    return "sent"


def run(dry_run: bool = False) -> str:
    title, body, state = build()
    status = "dry run" if dry_run else send(title, body)
    if not dry_run:
        store.put_json("digest_state", state)
    return f"{title}\n{body}\n[{status}]"
