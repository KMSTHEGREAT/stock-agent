"""
V3 Complete Analysis Pipeline.
Combines ALL analysis from V1, V2, and V3 into one unified workflow.

Runs:
1. V1 Technical Analysis (RSI, MACD, Bollinger, ATR)
2. V1 Fundamental Analysis (P/E, ROCE, promoter holding)
3. V3 DCF Valuation (intrinsic value)
4. V3 DuPont Analysis (ROE decomposition)
5. V3 CFA Risk Metrics (VaR, Sharpe, Beta)
6. V3 Multi-Source News (9 feeds)
7. LLM Bull/Bear Debate (using ALL above data)
8. Final Rating (Buy/OW/Hold/UW/Sell)
9. Record prediction for self-improvement
"""
import logging
import time
from typing import Optional, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class V3Pipeline:
    """Complete unified analysis pipeline."""
    
    def __init__(self):
        # V1 analyzers (technical + fundamental)
        self._tech_analyzer = None
        self._fund_analyzer = None
        self._tech_scorer = None
        self._fund_scorer = None
        
        # V3 CFA agents
        self._dcf_agent = None
        self._dupont_agent = None
        self._risk_agent = None
        
        # V3 news
        self._news_agg = None
        
        # Self-improvement
        self._si_engine = None
        
        # LLM
        self._has_llm = False
    
    def _get_tech_analyzer(self):
        if self._tech_analyzer is None:
            from src.v1.analysis.technical import TechnicalAnalyzer
            from src.v1.analysis.technical_scorer import TechnicalScorer
            self._tech_analyzer = TechnicalAnalyzer()
            self._tech_scorer = TechnicalScorer()
        return self._tech_analyzer
    
    def _get_fund_analyzer(self):
        if self._fund_analyzer is None:
            from src.v1.analysis.fundamental import FundamentalAnalyzer
            from src.v1.analysis.fundamental_scorer import FundamentalScorer
            self._fund_analyzer = FundamentalAnalyzer()
            self._fund_scorer = FundamentalScorer()
        return self._fund_analyzer
    
    def _get_dcf_agent(self):
        if self._dcf_agent is None:
            from src.v3.agents.dcf_valuation import DCFValuationAgent
            self._dcf_agent = DCFValuationAgent()
        return self._dcf_agent
    
    def _get_dupont_agent(self):
        if self._dupont_agent is None:
            from src.v3.agents.dupont_analyzer import DuPontAnalyzer
            self._dupont_agent = DuPontAnalyzer()
        return self._dupont_agent
    
    def _get_risk_agent(self):
        if self._risk_agent is None:
            from src.v3.agents.risk_manager_cfa import AdvancedRiskManager
            self._risk_agent = AdvancedRiskManager()
        return self._risk_agent
    
    def _get_news_agg(self):
        if self._news_agg is None:
            from src.v3.news.multi_source import MultiSourceNewsAggregator
            self._news_agg = MultiSourceNewsAggregator()
        return self._news_agg
    
    def _get_si_engine(self):
        if self._si_engine is None:
            from src.v3.selfimprove.engine import SelfImprovementEngine
            self._si_engine = SelfImprovementEngine()
        return self._si_engine
    
    def analyze(self, symbol: str, include_llm: bool = True,
               include_news: bool = True) -> Dict:
        """
        Run complete analysis for a stock.
        
        Returns dict with all results:
        - technical: V1 technical analysis + score
        - fundamental: V1 fundamental analysis + score
        - dcf: V3 DCF valuation
        - dupont: V3 DuPont decomposition
        - risk: V3 CFA risk metrics
        - news: V3 multi-source news + sentiment
        - debate: LLM bull/bear debate results
        - rating: Final rating (Buy/OW/Hold/UW/Sell)
        - overall_score: 0-100 composite
        - entry/target/stop: Price levels
        """
        symbol = symbol.upper()
        start_time = time.time()
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # ============ 1. TECHNICAL ANALYSIS (V1) ============
        logger.info(f"[V3 Pipeline] Technical analysis for {symbol}")
        try:
            tech_analyzer = self._get_tech_analyzer()
            tech_metrics = tech_analyzer.analyze(symbol, period="1y")
            
            if tech_metrics:
                from src.v1.analysis.technical_scorer import TechnicalScorer
                tech_scores = TechnicalScorer().score(tech_metrics)
                result["technical"] = {
                    "metrics": tech_metrics.model_dump(),
                    "score": tech_scores.get("total", 50),
                    "rsi": tech_metrics.rsi,
                    "trend": tech_metrics.trend.value if tech_metrics.trend else "neutral",
                    "current_price": tech_metrics.current_price,
                    "support": tech_metrics.support_level,
                    "resistance": tech_metrics.resistance_level,
                    "atr": tech_metrics.atr,
                }
            else:
                result["technical"] = None
        except Exception as e:
            logger.warning(f"[V3 Pipeline] Technical analysis failed: {e}")
            result["technical"] = None
        
        # ============ 2. FUNDAMENTAL ANALYSIS (V1) ============
        logger.info(f"[V3 Pipeline] Fundamental analysis for {symbol}")
        try:
            fund_analyzer = self._get_fund_analyzer()
            fund_metrics = fund_analyzer.analyze(symbol)
            
            if fund_metrics:
                from src.v1.analysis.fundamental_scorer import FundamentalScorer
                fund_scores = FundamentalScorer().score(fund_metrics)
                result["fundamental"] = {
                    "metrics": fund_metrics.model_dump() if hasattr(fund_metrics, 'model_dump') else {},
                    "score": fund_scores.get("total", 50),
                    "pe_ratio": fund_metrics.pe_ratio,
                    "roce": fund_metrics.roce,
                    "promoter_holding": fund_metrics.promoter_holding,
                    "pledged_pct": fund_metrics.pledged_pct,
                }
            else:
                result["fundamental"] = None
        except Exception as e:
            logger.warning(f"[V3 Pipeline] Fundamental analysis failed: {e}")
            result["fundamental"] = None
        
        # ============ 3. DCF VALUATION (V3) ============
        logger.info(f"[V3 Pipeline] DCF valuation for {symbol}")
        try:
            dcf = self._get_dcf_agent().analyze(symbol)
            result["dcf"] = {
                "intrinsic_value": dcf.intrinsic_value_per_share if dcf else None,
                "current_price": dcf.current_price if dcf else None,
                "upside_pct": dcf.upside_downside_pct if dcf else None,
                "status": dcf.valuation_status if dcf else None,
                "wacc": dcf.wacc if dcf else None,
            } if dcf else None
        except Exception as e:
            logger.warning(f"[V3 Pipeline] DCF failed: {e}")
            result["dcf"] = None
        
        # ============ 4. DUPONT ANALYSIS (V3) ============
        logger.info(f"[V3 Pipeline] DuPont for {symbol}")
        try:
            dupont = self._get_dupont_agent().analyze(symbol)
            result["dupont"] = {
                "roe": dupont.roe if dupont else None,
                "primary_driver": dupont.primary_driver if dupont else None,
                "quality_score": dupont.roe_quality_score if dupont else None,
                "operating_margin": dupont.operating_margin if dupont else None,
                "leverage": dupont.financial_leverage if dupont else None,
            } if dupont else None
        except Exception as e:
            logger.warning(f"[V3 Pipeline] DuPont failed: {e}")
            result["dupont"] = None
        
        # ============ 5. RISK ANALYSIS (V3) ============
        logger.info(f"[V3 Pipeline] Risk analysis for {symbol}")
        try:
            risk = self._get_risk_agent().analyze(symbol)
            result["risk"] = {
                "risk_category": risk.risk_category if risk else None,
                "risk_score": risk.risk_score if risk else None,
                "var_95": risk.var_95 if risk else None,
                "sharpe": risk.sharpe_ratio if risk else None,
                "beta": risk.beta if risk else None,
                "max_drawdown": risk.max_drawdown if risk else None,
                "position_pct": risk.recommended_position_pct if risk else None,
            } if risk else None
        except Exception as e:
            logger.warning(f"[V3 Pipeline] Risk analysis failed: {e}")
            result["risk"] = None
        
        # ============ 6. NEWS ANALYSIS (V3) ============
        news_sentiment = None
        if include_news:
            logger.info(f"[V3 Pipeline] News analysis for {symbol}")
            try:
                news_agg = self._get_news_agg()
                news_items = news_agg.get_stock_news(symbol)
                
                if news_items:
                    news_sentiment = news_agg.detect_sentiment_keywords(news_items)
                    diversity = news_agg.analyze_source_diversity(news_items)
                    
                    result["news"] = {
                        "sentiment_score": news_sentiment.get("score", 50),
                        "sentiment_label": news_sentiment.get("label", "neutral"),
                        "item_count": len(news_items),
                        "diversity_score": diversity.get("diversity_score", 0),
                        "items": [
                            {
                                "title": item.title,
                                "source": item.source,
                                "sentiment": item.sentiment_hint,
                            } for item in news_items[:10]
                        ],
                    }
                else:
                    result["news"] = None
            except Exception as e:
                logger.warning(f"[V3 Pipeline] News analysis failed: {e}")
                result["news"] = None
        
        # ============ 7. LLM DEBATE ============
        if include_llm:
            logger.info(f"[V3 Pipeline] LLM debate for {symbol}")
            try:
                debate = self._run_debate(symbol, result, news_sentiment)
                result["debate"] = debate
            except Exception as e:
                logger.warning(f"[V3 Pipeline] LLM debate failed: {e}")
                result["debate"] = None
        
        # ============ 8. FINAL RATING ============
        rating_result = self._compute_final_rating(result)
        result["rating"] = rating_result["rating"]
        result["overall_score"] = rating_result["score"]
        result["conviction"] = rating_result["conviction"]
        result["entry_price"] = rating_result.get("entry")
        result["target_price"] = rating_result.get("target")
        result["stop_loss"] = rating_result.get("stop")
        
        # ============ 9. RECORD PREDICTION ============
        try:
            si_engine = self._get_si_engine()
            si_engine.record_prediction(
                symbol=symbol,
                rating=result["rating"],
                score=result["overall_score"],
                target=result["target_price"],
                conviction=result["conviction"],
            )
        except Exception as e:
            logger.debug(f"[V3 Pipeline] Prediction recording failed: {e}")
        
        # Processing time
        result["processing_time"] = round(time.time() - start_time, 1)
        
        return result
    
    def _run_debate(self, symbol: str, analysis: Dict, news_sentiment) -> Optional[Dict]:
        """Run LLM debate with all analysis context."""
        try:
            from src.v2.llm import call_llm, extract_json
        except ImportError:
            return None
        
        # Build comprehensive context
        context_parts = [f"STOCK: {symbol}"]
        
        # Technical
        tech = analysis.get("technical")
        if tech:
            context_parts.append(f"\nTECHNICAL (Score: {tech['score']:.0f}/100):")
            context_parts.append(f"  Price: ₹{tech['current_price']:,.0f}")
            context_parts.append(f"  RSI: {tech['rsi']}")
            context_parts.append(f"  Trend: {tech['trend']}")
        
        # Fundamental
        fund = analysis.get("fundamental")
        if fund:
            context_parts.append(f"\nFUNDAMENTAL (Score: {fund['score']:.0f}/100):")
            if fund.get('pe_ratio'):
                context_parts.append(f"  P/E: {fund['pe_ratio']:.1f}")
            if fund.get('roce'):
                context_parts.append(f"  ROCE: {fund['roce']:.1f}%")
            if fund.get('promoter_holding') is not None:
                context_parts.append(f"  Promoter: {fund['promoter_holding']:.0f}%")
        
        # DCF
        dcf = analysis.get("dcf")
        if dcf:
            context_parts.append(f"\nDCF VALUATION:")
            context_parts.append(f"  Intrinsic: ₹{dcf['intrinsic_value']:,.0f}")
            context_parts.append(f"  Upside: {dcf['upside_pct']:+.1f}%")
            context_parts.append(f"  Status: {dcf['status']}")
        
        # DuPont
        dupont = analysis.get("dupont")
        if dupont:
            context_parts.append(f"\nPROFITABILITY:")
            context_parts.append(f"  ROE: {dupont['roe']:.1f}%")
            context_parts.append(f"  Driver: {dupont['primary_driver']}")
            context_parts.append(f"  Quality: {dupont['quality_score']:.0f}/100")
        
        # Risk
        risk = analysis.get("risk")
        if risk:
            context_parts.append(f"\nRISK:")
            context_parts.append(f"  Category: {risk['risk_category']}")
            context_parts.append(f"  VaR: {risk['var_95']:.1f}%")
            context_parts.append(f"  Sharpe: {risk['sharpe']:.2f}")
        
        # News
        if news_sentiment:
            context_parts.append(f"\nNEWS: {news_sentiment.get('label', 'neutral')} "
                               f"({news_sentiment.get('score', 50)}/100)")
        
        context = "\n".join(context_parts)
        
        # Bull
        bull_prompt = f"""{context}

Build the strongest BULL case for {symbol}. Use ONLY the data above.

Respond with ONLY JSON:
{{"thesis": "2-3 sentences", "evidence": ["point 1", "point 2", "point 3"], "conviction": 0.7}}"""
        
        bull = extract_json(call_llm(bull_prompt, temperature=0.4, num_predict=400))
        
        # Bear
        bear_prompt = f"""{context}

Build the strongest BEAR case against {symbol}. Use ONLY the data above.

Respond with ONLY JSON:
{{"thesis": "2-3 sentences", "evidence": ["risk 1", "risk 2", "risk 3"], "conviction": 0.6}}"""
        
        bear = extract_json(call_llm(bear_prompt, temperature=0.4, num_predict=400))
        
        # Debate
        bull_thesis = bull.get("thesis", "N/A") if bull else "N/A"
        bear_thesis = bear.get("thesis", "N/A") if bear else "N/A"
        
        debate_prompt = f"""Evaluate this debate for {symbol}:

BULL: {bull_thesis}
BEAR: {bear_thesis}

Context: {context[:500]}

Respond with ONLY JSON:
{{"winner": "bull or bear or inconclusive", "reasoning": "1-2 sentences", "sentiment_score": 55}}"""
        
        debate = extract_json(call_llm(debate_prompt, temperature=0.2, num_predict=300))
        
        return {
            "bull_thesis": bull_thesis,
            "bull_evidence": bull.get("evidence", []) if bull else [],
            "bull_conviction": bull.get("conviction", 0.5) if bull else 0.5,
            "bear_thesis": bear_thesis,
            "bear_evidence": bear.get("evidence", []) if bear else [],
            "bear_conviction": bear.get("conviction", 0.5) if bear else 0.5,
            "winner": debate.get("winner", "inconclusive") if debate else "inconclusive",
            "reasoning": debate.get("reasoning", "") if debate else "",
            "sentiment_score": debate.get("sentiment_score", 50) if debate else 50,
        }
    
    def _compute_final_rating(self, analysis: Dict) -> Dict:
        """
        Compute final rating from all components.
        Combines V1 scores + V3 CFA + LLM debate.
        """
        scores = []
        weights = []
        
        # V1 Technical
        tech = analysis.get("technical")
        if tech:
            scores.append(tech["score"])
            weights.append(0.20)
        
        # V1 Fundamental
        fund = analysis.get("fundamental")
        if fund:
            scores.append(fund["score"])
            weights.append(0.25)
        
        # V3 DCF
        dcf = analysis.get("dcf")
        if dcf and dcf.get("upside_pct") is not None:
            # Convert upside to 0-100 score (50 = fair value)
            dcf_score = 50 + max(-25, min(25, dcf["upside_pct"] / 2))
            scores.append(dcf_score)
            weights.append(0.20)
        
        # V3 DuPont Quality
        dupont = analysis.get("dupont")
        if dupont and dupont.get("quality_score") is not None:
            scores.append(dupont["quality_score"])
            weights.append(0.10)
        
        # V3 Risk (inverted: low risk = high score)
        risk = analysis.get("risk")
        if risk and risk.get("risk_score") is not None:
            scores.append(100 - risk["risk_score"])
            weights.append(0.10)
        
        # LLM Debate
        debate = analysis.get("debate")
        if debate and debate.get("sentiment_score") is not None:
            scores.append(debate["sentiment_score"])
            weights.append(0.15)
        
        # Weighted average
        if scores and weights:
            total = sum(s * w for s, w in zip(scores, weights))
            weight_sum = sum(weights)
            overall = total / weight_sum if weight_sum > 0 else 50
        else:
            overall = 50
        
        # Determine rating
        if overall >= 75:
            rating = "BUY"
        elif overall >= 62:
            rating = "OVERWEIGHT"
        elif overall >= 45:
            rating = "HOLD"
        elif overall >= 32:
            rating = "UNDERWEIGHT"
        else:
            rating = "SELL"
        
        # Conviction
        conviction = 0.5
        if debate:
            conviction += 0.1 if debate.get("winner") in ("bull", "bear") else -0.05
        if risk and risk.get("risk_score", 50) > 60:
            conviction -= 0.15
        conviction = max(0.1, min(0.95, conviction))
        
        # Price levels
        entry = target = stop = None
        if tech and tech.get("current_price"):
            price = tech["current_price"]
            atr = tech.get("atr") or price * 0.02
            
            if rating in ("BUY", "OVERWEIGHT"):
                entry = round(price * 0.98, 2)
                target = round(price + atr * 2.5, 2)
                stop = round(price - atr * 1.5, 2)
            elif rating == "HOLD":
                entry = round(price * 0.94, 2)
                target = round(price + atr * 2.0, 2)
                stop = round(price - atr * 1.5, 2)
        
        return {
            "rating": rating,
            "score": round(overall, 1),
            "conviction": round(conviction, 2),
            "entry": entry,
            "target": target,
            "stop": stop,
        }


# Convenience function
def analyze_stock_v3(symbol: str, include_llm: bool = True,
                    include_news: bool = True) -> Dict:
    """Run complete V3 analysis."""
    pipeline = V3Pipeline()
    return pipeline.analyze(symbol, include_llm, include_news)