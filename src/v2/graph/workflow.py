"""
Complete LangGraph workflow for V2.
Orchestrates all agents: Analysts → Researchers → Debate → Risk → Portfolio Manager.
"""
import logging
import time
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

from src.v2.graph.state import AnalysisState
from src.v2.agents.technical_analyst import technical_analyst_node
from src.v2.agents.fundamental_analyst import fundamental_analyst_node
from src.v2.agents.news_analyst import news_analyst_node
from src.v2.agents.sentiment_analyst import sentiment_analyst_node
from src.v2.agents.bull_researcher import bull_researcher_node
from src.v2.agents.bear_researcher import bear_researcher_node
from src.v2.agents.debate_moderator import debate_moderator_node
from src.v2.agents.risk_manager import risk_manager_node
from src.v2.agents.portfolio_manager import portfolio_manager_node

logger = logging.getLogger(__name__)


def build_analysis_workflow():
    """
    Build the complete V2 analysis workflow.
    
    Architecture:
    1. Analysts run (technical, fundamental, news, sentiment)
    2. Bull and Bear researchers evaluate the data
    3. Debate moderator weighs arguments
    4. Risk manager assesses risk
    5. Portfolio manager makes final decision
    """
    
    workflow = StateGraph(AnalysisState)
    
    # Add all nodes
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("fundamental_analyst", fundamental_analyst_node)
    workflow.add_node("news_analyst", news_analyst_node)
    workflow.add_node("sentiment_analyst", sentiment_analyst_node)
    workflow.add_node("bull_researcher", bull_researcher_node)
    workflow.add_node("bear_researcher", bear_researcher_node)
    workflow.add_node("debate_moderator", debate_moderator_node)
    workflow.add_node("risk_manager", risk_manager_node)
    workflow.add_node("portfolio_manager", portfolio_manager_node)
    
    # Set entry point — technical analyst starts
    workflow.set_entry_point("technical_analyst")
    
    # Analysts run sequentially (they depend on each other for sentiment)
    workflow.add_edge("technical_analyst", "fundamental_analyst")
    workflow.add_edge("fundamental_analyst", "news_analyst")
    workflow.add_edge("news_analyst", "sentiment_analyst")
    
    # Sentiment analyst completes → run bull and bear researchers
    workflow.add_edge("sentiment_analyst", "bull_researcher")
    workflow.add_edge("sentiment_analyst", "bear_researcher")
    
    # Both researchers → debate moderator
    workflow.add_edge("bull_researcher", "debate_moderator")
    workflow.add_edge("bear_researcher", "debate_moderator")
    
    # Debate → risk → final decision
    workflow.add_edge("debate_moderator", "risk_manager")
    workflow.add_edge("risk_manager", "portfolio_manager")
    
    # End
    workflow.add_edge("portfolio_manager", END)
    
    return workflow.compile()


def analyze_stock(symbol: str, company_name: str = None) -> dict:
    """
    Run the complete V2 analysis workflow for a stock.
    
    Returns the final state with all outputs.
    """
    start_time = time.time()
    
    logger.info(f"{'='*60}")
    logger.info(f"V2 Analysis: {symbol}")
    logger.info(f"{'='*60}")
    
    # Build workflow
    graph = build_analysis_workflow()
    
    # Initial state
    initial_state = {
        "symbol": symbol,
        "company_name": company_name or symbol,
        "technical_output": None,
        "fundamental_output": None,
        "news_output": None,
        "sentiment_output": None,
        "bull_thesis": None,
        "bear_thesis": None,
        "debate_result": None,
        "risk_assessment": None,
        "final_recommendation": None,
        "errors": [],
        "warnings": [],
        "agent_logs": [],
        "start_time": datetime.now(timezone.utc).isoformat(),
        "current_step": "starting",
    }
    
    # Run the workflow
    result = graph.invoke(initial_state)
    
    # Add processing time
    result["processing_time_seconds"] = round(time.time() - start_time, 1)
    
    logger.info(f"V2 Analysis complete: {symbol} in {result['processing_time_seconds']}s")
    
    return result


def analyze_watchlist(symbols: list) -> list:
    """Analyze multiple stocks."""
    results = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] {symbol}")
        print("-" * 40)
        
        result = analyze_stock(symbol)
        if result.get("final_recommendation"):
            results.append(result)
    
    # Sort by overall score (highest first)
    results.sort(
        key=lambda x: x.get("final_recommendation", None).overall_score 
        if x.get("final_recommendation") else 0,
        reverse=True
    )
    
    return results