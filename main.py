"""
CLI interface for the stock analysis agent.
"""
import click
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import StockAnalysisPipeline
from config.settings import DATABASE, DISCLAIMER


def format_inr(value):
    """Format value as INR."""
    if value is None:
        return "N/A"
    return f"₹{value:,.2f}"


def print_report(result: Dict):
    """Print formatted analysis report."""
    symbol = result["symbol"]
    company = result.get("company_name", symbol)
    scores = result.get("scores", {})
    overall = result.get("overall_score", 0)
    rec = result.get("recommendation", "HOLD").upper()
    
    print("\n" + "=" * 60)
    print(f"  {symbol} - {company}")
    print(f"  {result.get('analysis_timestamp', '')[:10]}")
    print("=" * 60)
    
    # Score bar
    bar_len = 30
    filled = int(overall / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    print(f"\n  SCORE: {overall:.1f}/100  [{bar}]")
    print(f"  RECOMMENDATION: {rec}")
    
    # Sub-scores
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │ Technical:    {scores.get('technical_score', 0):>6.1f}/100         │")
    print(f"  │ Fundamental:  {scores.get('fundamental_score', 0):>6.1f}/100         │")
    print(f"  │ News:         {scores.get('news_score', 0):>6.1f}/100         │")
    print(f"  │ Macro:        {scores.get('macro_score', 0):>6.1f}/100         │")
    print(f"  └─────────────────────────────────────────┘")
    
    # Technical snapshot
    tech = result.get("technical_metrics", {})
    if tech:
        print(f"\n  TECHNICAL:")
        print(f"    Price: {format_inr(tech.get('current_price'))}")
        print(f"    Trend: {tech.get('trend', 'N/A')}")
        print(f"    RSI: {tech.get('rsi', 'N/A')}")
        print(f"    ATR: {format_inr(tech.get('atr'))}")
    
    # Recommendation details
    rec_details = result.get("recommendation_details", {})
    if rec_details:
        print(f"\n  RECOMMENDATION:")
        print(f"    Strategy: {rec_details.get('entry_strategy', 'N/A')}")
        if rec_details.get('entry_price'):
            print(f"    Entry: {format_inr(rec_details['entry_price'])}")
        if rec_details.get('stop_loss'):
            print(f"    Stop Loss: {format_inr(rec_details['stop_loss'])}")
        if rec_details.get('target_price'):
            print(f"    Target: {format_inr(rec_details['target_price'])}")
    
    # LLM Analysis
    llm = result.get("llm_analysis", {})
    if llm:
        print(f"\n  LLM ANALYSIS:")
        print(f"    {llm.get('analysis_summary', 'N/A')}")
        print(f"    Confidence: {llm.get('confidence_level', 0):.0%}")
        
        if llm.get('key_risks'):
            print(f"\n  RISKS:")
            for risk in llm['key_risks'][:3]:
                print(f"    ⚠ {risk}")
        
        if llm.get('key_catalysts'):
            print(f"\n  CATALYSTS:")
            for cat in llm['key_catalysts'][:3]:
                print(f"    ✓ {cat}")
    
    # News
    news = result.get("news_analysis", {})
    if news:
        print(f"\n  NEWS: {news.get('news_score', 50)}/100 ({news.get('sentiment_label', 'N/A')})")
    
    print(f"\n  Time: {result.get('processing_time_seconds', 0):.1f}s")
    print(f"\n{'=' * 60}")
    print(DISCLAIMER)


def print_watchlist_summary(results: List[Dict]):
    """Print watchlist analysis summary."""
    print("\n" + "=" * 70)
    print("  WATCHLIST ANALYSIS SUMMARY (Sorted by Score)")
    print("=" * 70)
    
    if not results:
        print("  No results.")
        return
    
    print(f"\n  {'#':<3} {'Symbol':<12} {'Score':<7} {'Rec':<10} {'Price':<12}")
    print(f"  {'─'*3} {'─'*12} {'─'*7} {'─'*10} {'─'*12}")
    
    for i, r in enumerate(results, 1):
        tech = r.get("technical_metrics", {})
        price = tech.get("current_price", 0)
        
        print(f"  {i:<3} {r['symbol']:<12} {r.get('overall_score', 0):<7.1f} "
              f"{r.get('recommendation', 'N/A').upper():<10} "
              f"₹{price:<11.2f}")
    
    print(f"\n{'=' * 70}")
    print(f"  Total: {len(results)} stocks analyzed")
    print(DISCLAIMER)


@click.group()
def cli():
    """Indian Stock Analysis Agent - Personal Research Tool"""
    pass


@cli.command()
@click.argument('symbol')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed output')
@click.option('--no-news', is_flag=True, help='Skip news analysis')
@click.option('--no-llm', is_flag=True, help='Skip LLM analysis')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def analyze(symbol, detailed, no_news, no_llm, as_json):
    """Analyze a single stock."""
    pipeline = StockAnalysisPipeline()
    
    result = pipeline.analyze_stock(
        symbol.upper(),
        include_news=not no_news,
        include_llm=not no_llm
    )
    
    if result is None:
        click.echo(f"Failed to analyze {symbol}", err=True)
        sys.exit(1)
    
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


@cli.command()
@click.argument('symbols', nargs=-1)
@click.option('--file', '-f', type=click.Path(), help='File with symbols (one per line)')
def watchlist(symbols, file):
    """Analyze multiple stocks."""
    all_symbols = list(symbols)
    
    if file and Path(file).exists():
        with open(file) as f:
            all_symbols.extend(line.strip() for line in f if line.strip())
    
    if not all_symbols:
        # Use saved watchlist
        pipeline = StockAnalysisPipeline()
        saved = pipeline.get_watchlist()
        all_symbols = [w["symbol"] for w in saved]
    
    if not all_symbols:
        click.echo("No symbols provided.", err=True)
        sys.exit(1)
    
    # Remove duplicates
    seen = set()
    unique = []
    for s in all_symbols:
        s = s.upper().strip()
        if s not in seen:
            seen.add(s)
            unique.append(s)
    
    pipeline = StockAnalysisPipeline()
    results = pipeline.analyze_watchlist(unique)
    
    if results:
        print_watchlist_summary(results)
    else:
        click.echo("No results generated.", err=True)


@cli.command()
@click.argument('symbol')
def add(symbol):
    """Add symbol to watchlist."""
    pipeline = StockAnalysisPipeline()
    pipeline.add_to_watchlist(symbol.upper())
    click.echo(f"✓ Added {symbol.upper()} to watchlist")


@cli.command()
@click.argument('symbol')
def remove(symbol):
    """Remove symbol from watchlist."""
    pipeline = StockAnalysisPipeline()
    pipeline.remove_from_watchlist(symbol.upper())
    click.echo(f"✓ Removed {symbol.upper()} from watchlist")


@cli.command()
def list_watchlist():
    """List watchlist symbols."""
    pipeline = StockAnalysisPipeline()
    items = pipeline.get_watchlist()
    
    if items:
        click.echo("Watchlist:")
        for item in items:
            name = f" - {item['company_name']}" if item['company_name'] else ""
            click.echo(f"  {item['symbol']}{name}")
    else:
        click.echo("Watchlist is empty.")


@cli.command()
def market():
    """Show market context."""
    from src.analysis.market_context import MarketContextAnalyzer
    
    analyzer = MarketContextAnalyzer()
    context = analyzer.get_market_context()
    score = analyzer.get_macro_score(context)
    
    click.echo("\n" + "=" * 40)
    click.echo("  INDIAN MARKET CONTEXT")
    click.echo("=" * 40)
    click.echo(f"  Regime: {context.market_regime.upper()}")
    click.echo(f"  Nifty: {context.nifty_trend.value.upper()}")
    click.echo(f"  Sensex: {context.sensex_trend.value.upper()}")
    if context.vix_level:
        click.echo(f"  VIX: {context.vix_level}")
    click.echo(f"  Macro Score: {score:.0f}/100")
    click.echo("=" * 40)


@cli.command()
@click.option('--symbol', '-s', help='Filter by symbol')
@click.option('--limit', '-l', default=10, help='Number of results')
def history(symbol, limit):
    """Show analysis history."""
    import sqlite3
    
    conn = sqlite3.connect(DATABASE["path"])
    
    if symbol:
        rows = conn.execute("""
            SELECT symbol, overall_score, recommendation, current_price, analysis_timestamp
            FROM analysis_results WHERE symbol = ?
            ORDER BY analysis_timestamp DESC LIMIT ?
        """, (symbol.upper(), limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT symbol, overall_score, recommendation, current_price, analysis_timestamp
            FROM analysis_results
            ORDER BY analysis_timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
    
    conn.close()
    
    if not rows:
        click.echo("No history found.")
        return
    
    click.echo("\n" + "=" * 60)
    click.echo("  ANALYSIS HISTORY")
    click.echo("=" * 60)
    
    for row in rows:
        date = row[4][:10] if row[4] else "N/A"
        click.echo(f"  {date} | {row[0]:<10} | {row[1]:>5.1f} | {row[2].upper():<10} | ₹{row[3] or 0:>10.2f}")
    
    click.echo("=" * 60)


@cli.command()
def clear_cache():
    """Clear all cached data."""
    from src.data.cache import DataCache
    cache = DataCache()
    cache.clear_all()
    click.echo("✓ Cache cleared")


if __name__ == '__main__':
    cli()