"""
V2 CLI interface using LangGraph workflow.
"""
import click
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.v2.graph.workflow import analyze_stock, analyze_watchlist


def print_v2_report(result: dict):
    """Print V2 analysis report."""
    rec = result.get("final_recommendation")
    if not rec:
        print("No recommendation generated.")
        return
    
    debate = result.get("debate_result")
    risk = result.get("risk_assessment")
    bull = result.get("bull_thesis")
    bear = result.get("bear_thesis")
    tech = result.get("technical_output")
    
    print("\n" + "=" * 60)
    print(f"  {rec.symbol} — {rec.company_name}")
    print(f"  V2 Analysis • {rec.analysis_timestamp[:10]}")
    print("=" * 60)
    
    # Rating with color coding
    rating_colors = {
        "buy": "\033[92m", "overweight": "\033[92m",
        "hold": "\033[93m", "underweight": "\033[91m", "sell": "\033[91m"
    }
    color = rating_colors.get(rec.rating.value, "")
    reset = "\033[0m"
    
    print(f"\n  RATING: {color}{rec.rating.value.upper()}{reset}")
    print(f"  SCORE: {rec.overall_score:.1f}/100")
    print(f"  CONVICTION: {rec.conviction:.0%}")
    
    # Price levels
    if rec.entry_price:
        print(f"\n  ENTRY: ₹{rec.entry_price:,.2f}")
    if rec.target_price:
        print(f"  TARGET: ₹{rec.target_price:,.2f}")
    if rec.stop_loss:
        print(f"  STOP LOSS: ₹{rec.stop_loss:,.2f}")
    if rec.risk_reward_ratio:
        print(f"  RISK/REWARD: 1:{rec.risk_reward_ratio}")
    
    # Debate summary
    print(f"\n  {'─'*50}")
    print(f"  DEBATE ANALYSIS")
    print(f"  {'─'*50}")
    if debate:
        print(f"  Winner: {debate.winner.upper()}")
        print(f"  Sentiment: {debate.sentiment_score:.0f}/100")
    
    if bull:
        print(f"\n  🐂 BULL: {bull.thesis[:200]}")
    if bear:
        print(f"\n  🐻 BEAR: {bear.thesis[:200]}")
    
    # Risk warnings
    if risk and risk.risk_warnings:
        print(f"\n  {'─'*50}")
        print(f"  RISK WARNINGS")
        print(f"  {'─'*50}")
        for warning in risk.risk_warnings[:4]:
            print(f"  {warning}")
    
    # LLM Summary
    if rec.debate_summary:
        print(f"\n  {'─'*50}")
        print(f"  SUMMARY")
        print(f"  {'─'*50}")
        print(f"  {rec.debate_summary}")
    
    # Analyst scores
    print(f"\n  {'─'*50}")
    print(f"  COMPONENT SCORES")
    print(f"  {'─'*50}")
    
    if tech:
        print(f"  Technical:   {tech.score:.1f}/100")
    fund = result.get("fundamental_output")
    if fund:
        print(f"  Fundamental: {fund.score:.1f}/100")
    news = result.get("news_output")
    if news:
        print(f"  News:        {news.score:.1f}/100")
    if debate:
        print(f"  Debate:      {debate.sentiment_score:.1f}/100")
    if risk:
        print(f"  Risk:        {risk.risk_score:.1f}/100 (inverted)")
    
    # Errors
    if result.get("errors"):
        print(f"\n  ⚠ Errors: {result['errors']}")
    
    print(f"\n  Processing time: {result.get('processing_time_seconds', 0)}s")
    print(f"\n{'='*60}")


@click.group()
def cli():
    """V2 Stock Analysis Agent — Multi-Agent LangGraph Workflow"""
    pass


@cli.command()
@click.argument('symbol')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
def analyze(symbol, as_json):
    """Analyze a stock using the V2 multi-agent workflow."""
    result = analyze_stock(symbol.upper())
    
    if as_json:
        # Convert Pydantic objects to dicts for JSON
        output = _serialize_result(result)
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        print_v2_report(result)


@cli.command()
@click.argument('symbols', nargs=-1)
def watchlist(symbols):
    """Analyze multiple stocks."""
    if not symbols:
        click.echo("Provide symbols: main_v2.py watchlist RELIANCE TCS INFY")
        return
    
    results = analyze_watchlist([s.upper() for s in symbols])
    
    # Print summary table
    print("\n" + "=" * 70)
    print("  V2 WATCHLIST ANALYSIS (Sorted by Score)")
    print("=" * 70)
    print(f"\n  {'Symbol':<12} {'Rating':<12} {'Score':<7} {'Conviction':<11}")
    print(f"  {'─'*12} {'─'*12} {'─'*7} {'─'*11}")
    
    for result in results:
        rec = result.get("final_recommendation")
        if rec:
            print(f"  {rec.symbol:<12} {rec.rating.value.upper():<12} "
                  f"{rec.overall_score:<7.1f} {rec.conviction:<11.0%}")
    
    print(f"\n{'='*70}")


def _serialize_result(result: dict) -> dict:
    """Convert result to JSON-serializable format."""
    output = {}
    for key, value in result.items():
        if hasattr(value, "model_dump"):
            output[key] = value.model_dump()
        elif isinstance(value, list) and value and hasattr(value[0], "model_dump"):
            output[key] = [v.model_dump() for v in value]
        else:
            output[key] = value
    return output


if __name__ == '__main__':
    cli()