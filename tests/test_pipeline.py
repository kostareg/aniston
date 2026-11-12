"""End-to-end checks on the synthetic league from conftest: projections, perceived value, trades, the season
simulation around the current week and the playoffs, and everything `analyze` stores."""
import numpy as np
import polars as pl

from aniston import store


def test_open_week_counts_until_kickoff(league):
    from aniston.model.projections import build

    league(current_week=5, started=False)
    assert build().weeks == [5, 6, 7, 8, 9, 10]
    league(current_week=5, started=True)
    proj = build()
    assert proj.weeks == [6, 7, 8, 9, 10] and proj.cur_week == 5 and proj.ens.shape[1] == 5


def test_free_agent_pool_ranks_by_best_source_not_espn(league):
    from aniston.model.projections import build

    league()
    # One free agent per position only: the RB and WR that survive must be the consensus-best, and ESPN
    # underrates even ids, so ranking by ESPN alone would keep the wrong one where the two disagree.
    proj = build(fa_per_pos=1)
    fa = proj.players.filter(pl.col("fantasy_team_id") == 0)
    all_fa = store.read("players").filter(pl.col("fantasy_team_id") == 0)
    truth = store.read("sleeper_weeks").group_by("sleeper_id").agg(pl.col("sleeper_proj").sum())
    for pos in ("RB", "WR"):
        ids = all_fa.filter(pl.col("pos") == pos)["espn_id"].to_list()
        best = max(ids, key=lambda e: truth.filter(pl.col("sleeper_id") == f"s{e}")["sleeper_proj"][0])
        assert best in fa.filter(pl.col("pos") == pos)["espn_id"].to_list()


def test_ensemble_sits_between_sources(league):
    from aniston.model.projections import build

    league()
    proj = build()
    lo = np.minimum(proj.espn, np.nan_to_num(proj.sources["sleeper"]))
    hi = np.maximum(proj.espn, np.nan_to_num(proj.sources["sleeper"]))
    assert ((proj.ens >= lo - 2.0) & (proj.ens <= hi + 2.0)).all()  # usage model may pull slightly outside
    assert (proj.ens.sum(1) > 0).all()


def test_perceived_value_follows_espn_not_consensus(league):
    from aniston.model.projections import build
    from aniston.trade.perceived import perceived_values

    league()
    proj = build()
    pv = perceived_values(proj)
    assert pv.height == proj.players.height and pv["pv"].is_finite().all()
    assert (pv["games_left"] > 0).all()
    # Same position and depth on one roster, but ESPN shows the even id 20% lower: he must look worse.
    team1 = pv.filter(pl.col("fantasy_team_id") == 1, pl.col("pos") == "QB").sort("espn_id")
    assert team1["espn_ros"][1] < team1["espn_ros"][0]


def test_build_trade_is_legal_and_consistent(league):
    from aniston.trade.engine import TradeContext

    league()
    ctx = TradeContext()
    mine, theirs = ctx.base[ctx.me], ctx.base[1]
    give = next(i for i in mine if ctx.pos[i] == "RB")
    get = next(i for i in theirs if ctx.pos[i] == "RB")
    t = ctx.evaluate(ctx.build_trade(1, (give,), (get,)))
    assert get in t.my_roster and give not in t.my_roster and give in t.their_roster and get not in t.their_roster
    assert len(t.my_roster) == len(mine) and len(t.their_roster) == len(theirs)
    assert np.isclose(t.my_gain, ctx.true.value(t.my_roster) - ctx.true.value(mine))
    assert 0 < t.p_accept < 1 and 0 < t.p_no_veto <= 1
    assert np.isclose(t.ev, t.d_title * t.p_accept * t.p_no_veto)


def test_target_search_only_returns_packages_with_the_target(league):
    from aniston.trade.engine import TradeContext

    league()
    ctx = TradeContext()
    target = max(ctx.base[1], key=lambda i: ctx.true.ros[i] if ctx.pos[i] == "RB" else -1)
    found = ctx.candidates(must_get={target})
    assert found and all(target in t.get and t.partner == 1 for t in found)


def test_partner_situation_moves_acceptance(league):
    from aniston.trade.acceptance import p_accept, partner_week_weights

    league()
    base, is_playoff = np.array([1.0, 1.0, 1.5]), np.array([False, False, True])
    assert np.allclose(partner_week_weights(base, is_playoff, 1.0), base)  # locked in: neutral
    bubble = partner_week_weights(base, is_playoff, 0.5)
    assert bubble[0] > 1.0 and bubble[2] < 1.5  # wants points now, discounts playoff weeks
    assert p_accept(5.0, 5.0, False, partner_p_playoffs=0.01) < p_accept(5.0, 5.0, False, partner_p_playoffs=0.5)


def test_playoff_week_in_progress_simulates(league):
    from aniston.model.projections import build
    from aniston.sim.season import Simulator, current_rosters

    # Week 10 is the final; the semi-finals (week 9: 1v4, 2v3, home scores 105/110/115/120) are in the books.
    league(current_week=10, started=True, deadline_in_days=-5)
    proj = build()
    assert proj.weeks == []
    sim = Simulator(proj)
    res = sim.run(current_rosters(proj))
    assert np.isclose(res.p_title.sum(), 1.0)
    # Seeds follow the regular season (team 1 best); week 9 scores rise with team id, so 4 beat 1 and 3 beat 2.
    assert res.p_title[sim.tix[1]] == 0 and res.p_title[sim.tix[2]] == 0
    assert res.p_title[sim.tix[3]] > res.p_title[sim.tix[4]] > 0  # 3 has the higher live projection


def test_analyze_stores_everything_and_snapshots(league):
    from aniston.analyze import analyze
    from aniston.digest import run as digest

    league()
    report = analyze()
    assert report["trades"] >= 0
    for table in ("sim_baseline", "player_model", "projection_weeks", "waivers", "streamers", "blocks",
                  "my_lineup", "start_sit", "snap_sim", "snap_proj"):
        assert store.has(table), table
    sim = store.read("sim_baseline")
    assert np.isclose(sim["p_title"].sum(), 1.0) and sim["p_win_next"].is_between(0, 1).all()
    assert (store.read("my_lineup")["slot"] != "").sum() == 4
    analyze()  # same day: the snapshot is replaced, not duplicated
    assert store.read("snap_sim").height == 4
    text = digest(dry_run=True)
    assert "Title" in text and "dry run" in text


def test_deadline_passed_skips_trade_search(league):
    from aniston.analyze import analyze

    league(deadline_in_days=-1)
    assert "skipped" in str(analyze()["trades"]) and not store.has("trade_ideas")


def test_offer_log_roundtrip(league):
    from aniston import offers
    from aniston.trade.engine import TradeContext

    league()
    ctx = TradeContext()
    give = next(i for i in ctx.base[ctx.me] if ctx.pos[i] == "WR")
    get = next(i for i in ctx.base[3] if ctx.pos[i] == "WR")
    t = ctx.evaluate(ctx.build_trade(3, (give,), (get,)))
    first, second = offers.log_offer(ctx, t, "first"), offers.log_offer(ctx, t)
    assert (first, second) == (1, 2) and offers.calibration() is None
    offers.set_status(first, "accepted")
    offers.set_status(second, "rejected", "said no")
    df = offers.list_offers()
    assert df.filter(pl.col("id") == 2)["note"][0] == "said no" and df["resolved"].null_count() == 0
    cal = offers.calibration()
    assert cal["n"] == 2 and cal["observed"] == 0.5 and cal["suggested_acc_bias"] is None
