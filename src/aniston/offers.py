"""Log of trades actually proposed and how they ended: the only real data on whether P(accept) is right.

Each offer stores the model's inputs at the time it was sent, so the acceptance curve can be checked (and its
bias re-centred) once enough offers are resolved. The `offers` table is never touched by refresh or analyze.
"""
import math
from datetime import datetime

import polars as pl

from aniston import store
from aniston.config import settings

STATUSES = ("proposed", "accepted", "rejected", "countered", "withdrawn")
RESOLVED = ("accepted", "rejected", "countered")  # a counter means the offer as sent was turned down
MIN_FOR_FIT = 8

SCHEMA = {
    "id": pl.Int64, "created": pl.Datetime, "week": pl.Int64, "partner_id": pl.Int64, "partner": pl.Utf8,
    "give": pl.Utf8, "get": pl.Utf8, "give_ids": pl.List(pl.Int64), "get_ids": pl.List(pl.Int64),
    "p_accept": pl.Float64, "p_no_veto": pl.Float64, "d_title": pl.Float64, "their_seen_gain": pl.Float64,
    "their_value_gain": pl.Float64, "partner_p_playoffs": pl.Float64, "partner_has_traded": pl.Boolean,
    "status": pl.Utf8, "resolved": pl.Datetime, "note": pl.Utf8,
}


def list_offers() -> pl.DataFrame:
    if not settings().db_path.exists() or not store.has("offers"):
        return pl.DataFrame(schema=SCHEMA)
    return store.read("offers").sort("id", descending=True)


def log_offer(ctx, t, note: str = "") -> int:
    """Record trade `t` (a scored `Trade` from `ctx`) as proposed. Returns the offer id."""
    existing = list_offers()
    new_id = int(existing["id"].max() or 0) + 1 if existing.height else 1
    eid = lambda ids: [int(ctx.players["espn_id"][i]) for i in ids]  # noqa: E731
    nm = lambda ids: " + ".join(ctx.names[i] for i in ids)  # noqa: E731
    row = {
        "id": new_id, "created": datetime.now(), "week": ctx.meta["current_week"], "partner_id": t.partner,
        "partner": ctx.teams[t.partner], "give": nm(t.give), "get": nm(t.get), "give_ids": eid(t.give),
        "get_ids": eid(t.get), "p_accept": t.p_accept, "p_no_veto": t.p_no_veto, "d_title": t.d_title,
        "their_seen_gain": t.their_seen_gain, "their_value_gain": t.their_pvor_gain,
        "partner_p_playoffs": t.partner_p_playoffs, "partner_has_traded": t.partner in ctx.has_traded,
        "status": "proposed", "resolved": None, "note": note,
    }
    store.append("offers", pl.DataFrame([row], schema=SCHEMA))
    return new_id


def set_status(offer_id: int, status: str, note: str | None = None) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    resolved = datetime.now() if status != "proposed" else None
    store.execute("UPDATE offers SET status = ?, resolved = ? WHERE id = ?", [status, resolved, offer_id])
    if note is not None:
        store.execute("UPDATE offers SET note = ? WHERE id = ?", [note, offer_id])


def calibration() -> dict | None:
    """Predicted vs observed acceptance over resolved offers, and the ACC_BIAS that would have matched it."""
    done = list_offers().filter(pl.col("status").is_in(RESOLVED))
    if not done.height:
        return None
    p = [min(max(x, 1e-6), 1 - 1e-6) for x in done["p_accept"].to_list()]
    observed = float((done["status"] == "accepted").mean())
    out = {"n": done.height, "predicted": sum(p) / len(p), "observed": observed, "suggested_acc_bias": None}
    if done.height >= MIN_FOR_FIT and 0 < observed < 1:
        logit = [math.log(x / (1 - x)) for x in p]
        mean_at = lambda d: sum(1 / (1 + math.exp(-(z + d))) for z in logit) / len(logit)  # noqa: E731
        lo, hi = -6.0, 6.0
        for _ in range(50):  # the shift in log-odds that makes the mean prediction equal the observed rate
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_at(mid) < observed else (lo, mid)
        out["suggested_acc_bias"] = round(settings().acc_bias + (lo + hi) / 2, 2)
    return out
