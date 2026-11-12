import logging

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main(verbose: bool = False):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")


@app.command()
def refresh():
    """Pull ESPN, Sleeper and nflverse data into the local database."""
    from aniston.refresh import refresh as run

    for k, v in run().items():
        typer.echo(f"{k:14} {v}")


@app.command()
def analyze(trades: bool = typer.Option(True, help="Also run the trade finder."),
            workers: int = typer.Option(0, help="Processes for the trade finder (0 = all CPU cores).")):
    """Run projections, the season simulation, waivers and the trade finder; store results for the dashboard."""
    import os

    from aniston.analyze import analyze as run

    typer.echo(run(with_trades=trades, workers=workers or os.cpu_count() or 1))


@app.command()
def reload(workers: int = typer.Option(0, help="Processes for the trade finder (0 = all CPU cores).")):
    """Refresh, then analyze: the whole manual reload in one command (what the dashboard's button runs)."""
    refresh()
    analyze(trades=True, workers=workers)


@app.command()
def backtest():
    """Score each projection source on completed weeks and fit ensemble weights."""
    import json

    from aniston.backtest import run

    typer.echo(json.dumps(run(), indent=2))


@app.command()
def sim():
    """Print playoff and title odds for every team."""
    from aniston import store

    df = store.read("sim_baseline").sort("p_title", descending=True)
    for r in df.iter_rows(named=True):
        typer.echo(f"{r['name'][:26]:26} wins {r['exp_wins']:5.2f}  playoffs {r['p_playoffs']:6.1%}  "
                   f"bye {r['p_bye']:6.1%}  title {r['p_title']:6.1%}")


@app.command()
def trades(n: int = typer.Option(10, "-n", "--n"), partner: str = typer.Option(None, help="Filter by partner team name substring.")):
    """Print the best trade ideas from the last analysis."""
    import polars as pl

    from aniston import store

    df = store.read("trade_ideas")
    if partner:
        df = df.filter(pl.col("partner").str.to_lowercase().str.contains(partner.lower()))
    _print_trades(df.head(n))


def _print_trades(df):
    for r in df.iter_rows(named=True):
        typer.echo(f"\n[{r['partner']}] give {r['give']}  ->  get {r['get']}")
        typer.echo(f"  title {r['d_title']:+.1%} (±{r['d_title_se']:.1%})  playoffs {r['d_playoffs']:+.1%}  "
                   f"P(accept) {r['p_accept']:.0%}  P(no veto) {r['p_no_veto']:.0%}  EV {r['ev']:+.2%}")
        typer.echo("  " + r["why"].replace(" | ", "\n  "))


@app.command()
def target(player: str = typer.Argument(..., help="Name (or part of it) of a player on another team."),
           n: int = typer.Option(8, "-n", "--n"),
           workers: int = typer.Option(0, help="Processes for the search (0 = all CPU cores).")):
    """Search for the best packages that land a specific player."""
    import os

    from aniston.trade.engine import TradeContext

    ctx = TradeContext()
    mine = set(ctx.base[ctx.me])
    hits = [i for r in ctx.base.values() for i in r if i not in mine and player.lower() in ctx.names[i].lower()]
    if not hits:
        typer.echo(f"No player matching '{player}' on another team's roster.")
        raise typer.Exit(1)
    typer.echo("Targeting " + ", ".join(ctx.names[i] for i in hits))
    found = ctx.find(must_get=set(hits), workers=workers or os.cpu_count() or 1)
    if not found:
        typer.echo("No package helps you and clears the acceptance threshold.")
        return
    _print_trades(ctx.frame(found).head(n))


@app.command()
def digest(dry_run: bool = typer.Option(False, help="Print the digest without sending or remembering it.")):
    """Send the post-refresh digest to NOTIFY_URL (ntfy topic or Discord webhook); prints it when unset."""
    from aniston.digest import run

    typer.echo(run(dry_run=dry_run))


@app.command()
def offers():
    """List the trades you have proposed and how they ended."""
    from aniston.offers import calibration, list_offers

    df = list_offers()
    if not df.height:
        typer.echo("No offers logged. Log one from the Trade Finder or Trade Analyzer page.")
        return
    for r in df.iter_rows(named=True):
        typer.echo(f"#{r['id']:<3} {r['created']:%b %d}  {r['status']:10} [{r['partner']}] give {r['give']} -> get "
                   f"{r['get']}  (P(accept) {r['p_accept']:.0%})" + (f"  {r['note']}" if r["note"] else ""))
    cal = calibration()
    if cal:
        typer.echo(f"\n{cal['n']} resolved: predicted {cal['predicted']:.0%} accepted, observed {cal['observed']:.0%}"
                   + (f"; ACC_BIAS={cal['suggested_acc_bias']} would match" if cal["suggested_acc_bias"] is not None else ""))


@app.command("offer-status")
def offer_status(offer_id: int, status: str, note: str = typer.Option(None)):
    """Mark an offer accepted, rejected, countered, withdrawn (or back to proposed)."""
    from aniston.offers import set_status

    try:
        set_status(offer_id, status, note)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from e
    typer.echo(f"offer #{offer_id}: {status}")


if __name__ == "__main__":
    app()
