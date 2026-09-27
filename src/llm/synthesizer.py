"""
LLM synthesis layer using Ollama with Qwen 3 4B.
The LLM interprets pre-computed metrics — it does NOT calculate scores.
"""
import json
import ollama
from typing import Optional, Dict, List
from datetime import datetime

from src.schemas.stock_analysis import (
    StockAnalysis, TechnicalMetrics, FundamentalMetrics,
    RiskFlags, Recommendation, MarketContext
)
from src.llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.utils.logger import setup_logger
from config.settings import LLM

logger = setup_logger(__name__)


class LLMSynthesizer:
    """Uses Qwen 3 4B to synthesize qualitative analysis from computed metrics."""
    
    def __init__(self, model: str = None):
        self.model = model or LLM["model"]
        self.client = ollama
    
    def synthesize_analysis(
        self,
        symbol: str,
        company_name: str,
        technical_metrics: TechnicalMetrics,
        technical_scores: Dict,
        fundamental_metrics: FundamentalMetrics,
        fundamental_scores: Dict,
        news_context: str = "",
        market_context: Optional[MarketContext] = None
    ) -> Optional[Dict]:
        """
        Generate qualitative analysis using LLM.
        
        Returns:
            Dictionary with LLM-generated fields:
            - analysis_summary
            - key_risks
            - key_catalysts
            - risk_flags
            - confidence_level
        """
        logger.info(f"Starting LLM synthesis for {symbol}")
        
        # Build the prompt
        prompt = self._build_prompt(
            symbol, company_name,
            technical_metrics, technical_scores,
            fundamental_metrics, fundamental_scores,
            news_context, market_context
        )
        
        # Define the output schema
        output_schema = {
            "type": "object",
            "properties": {
                "analysis_summary": {
                    "type": "string",
                    "description": "Brief 2-3 sentence analysis summary in Indian market context",
                    "maxLength": 500
                },
                "key_risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3-5 risks specific to Indian markets",
                    "maxItems": 5
                },
                "key_catalysts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3-5 potential positive catalysts",
                    "maxItems": 5
                },
                "risk_flags": {
                    "type": "object",
                    "properties": {
                        "promoter_pledge_risk": {
                            "type": "boolean",
                            "description": "High promoter pledging risk"
                        },
                        "fii_outflow_risk": {
                            "type": "boolean",
                            "description": "FII outflow pressure risk"
                        },
                        "liquidity_risk": {
                            "type": "boolean",
                            "description": "Low liquidity risk for small/mid caps"
                        },
                        "regulatory_risk": {
                            "type": "string",
                            "description": "Any regulatory concerns (or empty string)"
                        },
                        "corporate_governance_risk": {
                            "type": "string",
                            "description": "Governance concerns (or empty string)"
                        }
                    }
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence in analysis (0.0 to 1.0)",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": [
                "analysis_summary",
                "key_risks",
                "key_catalysts",
                "risk_flags",
                "confidence_level"
            ]
        }
        
        try:
            # Call Ollama
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                format=output_schema,
                options={
                    "temperature": LLM["temperature"],
                    "num_ctx": LLM["num_ctx"],
                    "num_predict": LLM["max_tokens"],
                }
            )
            
            # Parse response
            content = response['message']['content']
            result = json.loads(content)
            
            # Validate and sanitize
            result = self._validate_output(result)
            
            logger.info(f"LLM synthesis complete for {symbol}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {symbol}: {e}")
            return self._fallback_response(symbol)
        except Exception as e:
            logger.error(f"LLM error for {symbol}: {e}")
            return self._fallback_response(symbol)
    
    def analyze_news_sentiment(self, symbol: str, news_items: List[Dict]) -> Optional[str]:
        """Analyze sentiment from news items."""
        if not news_items:
            return "No recent news available."
        
        # Format news for prompt
        news_text = ""
        for i, item in enumerate(news_items[:5], 1):  # Limit to 5 items
            news_text += f"{i}. {item.get('title', 'No title')}\n"
            if item.get('summary'):
                news_text += f"   Summary: {item['summary'][:150]}\n"
            news_text += "\n"
        
        prompt = f"""Analyze these news items for {symbol}:

{news_text}

Provide a brief sentiment analysis (positive/negative/neutral) with key points. 
Keep it under 150 words."""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 300,
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
            return "News analysis unavailable."
    
    def _build_prompt(
        self,
        symbol: str,
        company_name: str,
        technical: TechnicalMetrics,
        tech_scores: Dict,
        fundamental: FundamentalMetrics,
        fund_scores: Dict,
        news_context: str,
        market_context: Optional[MarketContext]
    ) -> str:
        """Build the analysis prompt with all metrics."""
        
        # Interpret RSI
        rsi_interp = "Neutral"
        if technical.rsi > 70:
            rsi_interp = "Overbought - potential reversal risk"
        elif technical.rsi < 30:
            rsi_interp = "Oversold - potential bounce opportunity"
        elif technical.rsi > 60:
            rsi_interp = "Bullish momentum"
        elif technical.rsi < 40:
            rsi_interp = "Bearish momentum"
        
        # MACD status
        macd_status = "Neutral"
        if technical.macd is not None and technical.macd_signal is not None:
            if technical.macd > technical.macd_signal:
                macd_status = "Bullish (MACD above signal)"
            else:
                macd_status = "Bearish (MACD below signal)"
        
        # Market regime
        regime = market_context.market_regime if market_context else "neutral"
        
        prompt = USER_PROMPT_TEMPLATE.format(
            symbol=symbol,
            company_name=company_name or symbol,
            sector=fundamental.market_cap_category or "Unknown",
            
            technical_score=tech_scores.get('total', 50),
            current_price=technical.current_price,
            trend=technical.trend.value.upper(),
            rsi=technical.rsi,
            rsi_interpretation=rsi_interp,
            macd_signal_status=macd_status,
            atr=technical.atr or "N/A",
            support_level=technical.support_level or "N/A",
            resistance_level=technical.resistance_level or "N/A",
            price_change_1m=technical.price_change_pct_1m or 0,
            price_change_1w=technical.price_change_pct_1w or 0,
            volume_ratio=technical.volume_ratio or 1.0,
            
            fundamental_score=fund_scores.get('total', 50),
            pe_ratio=fundamental.pe_ratio or "N/A",
            pb_ratio=fundamental.pb_ratio or "N/A",
            roce=fundamental.roce or "N/A",
            roe=fundamental.roe or "N/A",
            debt_to_equity=fundamental.debt_to_equity or "N/A",
            promoter_holding=fundamental.promoter_holding or "N/A",
            pledged_pct=fundamental.pledged_pct or 0,
            market_cap_cr=fundamental.market_cap_cr or "N/A",
            market_cap_category=fundamental.market_cap_category.value if fundamental.market_cap_category else "N/A",
            revenue_growth=fundamental.revenue_growth_1y or "N/A",
            net_margin=fundamental.net_margin or "N/A",
            
            overall_score=(tech_scores.get('total', 50) + fund_scores.get('total', 50)) / 2,
            market_regime=regime,
            
            news_context=news_context or "No recent news data available."
        )
        
        return prompt
    
    def _validate_output(self, result: Dict) -> Dict:
        """Validate and sanitize LLM output."""
        # Ensure required fields exist
        if 'analysis_summary' not in result:
            result['analysis_summary'] = "Analysis summary unavailable."
        
        # Truncate summary if too long
        if len(result['analysis_summary']) > 500:
            result['analysis_summary'] = result['analysis_summary'][:497] + "..."
        
        # Ensure lists
        if not isinstance(result.get('key_risks'), list):
            result['key_risks'] = ["Unable to determine specific risks"]
        result['key_risks'] = [str(r) for r in result['key_risks'][:5]]
        
        if not isinstance(result.get('key_catalysts'), list):
            result['key_catalysts'] = ["Unable to determine specific catalysts"]
        result['key_catalysts'] = [str(c) for c in result['key_catalysts'][:5]]
        
        # Validate risk flags
        if 'risk_flags' not in result or not isinstance(result['risk_flags'], dict):
            result['risk_flags'] = {}
        
        rf = result['risk_flags']
        rf['promoter_pledge_risk'] = bool(rf.get('promoter_pledge_risk', False))
        rf['fii_outflow_risk'] = bool(rf.get('fii_outflow_risk', False))
        rf['liquidity_risk'] = bool(rf.get('liquidity_risk', False))
        rf['regulatory_risk'] = str(rf.get('regulatory_risk', '') or '')
        rf['corporate_governance_risk'] = str(rf.get('corporate_governance_risk', '') or '')
        
        # Validate confidence
        confidence = result.get('confidence_level', 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        result['confidence_level'] = max(0.0, min(1.0, float(confidence)))
        
        return result
    
    def _fallback_response(self, symbol: str) -> Dict:
        """Fallback response if LLM fails."""
        return {
            "analysis_summary": f"LLM analysis unavailable for {symbol}. Rely on quantitative scores only.",
            "key_risks": ["LLM analysis failed - unable to assess qualitative risks"],
            "key_catalysts": ["LLM analysis failed - unable to assess catalysts"],
            "risk_flags": {
                "promoter_pledge_risk": False,
                "fii_outflow_risk": False,
                "liquidity_risk": False,
                "regulatory_risk": "",
                "corporate_governance_risk": ""
            },
            "confidence_level": 0.3  # Low confidence when LLM fails
        }