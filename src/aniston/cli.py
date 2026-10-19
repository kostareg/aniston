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
def analyze(trades: bool = typer.Option(True, help="Also run the trade finder (about 30s).")):
    """Run projections, the season simulation, waivers and the trade finder; store results for the dashboard."""
    from aniston.analyze import analyze as run

    typer.echo(run(with_trades=trades))


@app.command()
def backtest():
    """Score each projection source on completed weeks and fit ensemble weights."""
    import json

    from aniston.backtest import run

    typer.echo(json.dumps(run(), indent=2))


@app.command()
def sim():
    """Print playoff and title odds for every team."""
    import polars as pl

    from aniston import store

    df = store.read("sim_baseline").sort("p_title", descending=True)
    for r in df.iter_rows(named=True):
        typer.echo(f"{r['name'][:26]:26} wins {r['exp_wins']:5.2f}  playoffs {r['p_playoffs']:6.1%}  "
                   f"bye {r['p_bye']:6.1%}  title {r['p_title']:6.1%}")
    _ = pl


@app.command()
def trades(n: int = typer.Option(10, "-n", "--n"), partner: str = typer.Option(None, help="Filter by partner team name substring.")):
    """Print the best trade ideas from the last analysis."""
    import polars as pl

    from aniston import store

    df = store.read("trade_ideas")
    if partner:
        df = df.filter(pl.col("partner").str.to_lowercase().str.contains(partner.lower()))
    for r in df.head(n).iter_rows(named=True):
        typer.echo(f"\n[{r['partner']}] give {r['give']}  ->  get {r['get']}")
        typer.echo(f"  title {r['d_title']:+.1%} (±{r['d_title_se']:.1%})  playoffs {r['d_playoffs']:+.1%}  "
                   f"P(accept) {r['p_accept']:.0%}  P(no veto) {r['p_no_veto']:.0%}  EV {r['ev']:+.2%}")
        typer.echo("  " + r["why"].replace(" | ", "\n  "))


if __name__ == "__main__":
    app()
