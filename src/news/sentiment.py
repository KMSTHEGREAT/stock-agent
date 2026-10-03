"""
News sentiment scorer using LLM and keyword fallback.
"""
import json
import re
import ollama
from typing import List, Dict, Optional

from src.news.aggregator import IndianNewsAggregator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NewsSentimentScorer:
    """Scores news sentiment for stocks."""
    
    def __init__(self, aggregator: Optional[IndianNewsAggregator] = None):
        self.aggregator = aggregator or IndianNewsAggregator()
    
    def score_news_sentiment(self, symbol: str, 
                            company_name: str = None) -> Dict:
        """Score news sentiment for a stock. Returns 0-100 score."""
        try:
            news_items = self.aggregator.get_stock_news(symbol, company_name)
        except Exception as e:
            logger.error(f"News fetch error for {symbol}: {e}")
            news_items = []
        
        if not news_items:
            return {
                "news_score": 50,
                "sentiment_label": "neutral",
                "summary": "No recent news found",
                "item_count": 0
            }
        
        # Try LLM analysis
        analysis = self._analyze_with_llm(symbol, news_items)
        if analysis:
            return analysis
        
        # Fallback to keyword analysis
        return self._keyword_fallback(news_items)
    
    def _analyze_with_llm(self, symbol: str, news_items: List[Dict]) -> Optional[Dict]:
        """Use LLM to analyze news sentiment."""
        news_text = ""
        for i, item in enumerate(news_items[:5], 1):
            news_text += f"{i}. {item.get('title', '')}\n"
            if item.get('summary'):
                news_text += f"   {item['summary'][:100]}\n"
        
        prompt = f"""Analyze news sentiment for {symbol}:

{news_text}

Respond with ONLY this JSON:
{{"sentiment_score": 50, "sentiment_label": "neutral", "summary": "brief analysis"}}"""
        
        try:
            response = ollama.chat(
                model='qwen3:4b',
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 500}
            )
            
            # Get content or thinking
            content = response.message.content or ""
            thinking = response.message.thinking or ""
            text = content if content.strip() else thinking
            
            if not text.strip():
                return None
            
            # Find JSON
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    
                    score = result.get('sentiment_score', 50)
                    if not isinstance(score, (int, float)):
                        score = 50
                    score = max(0, min(100, float(score)))
                    
                    return {
                        "news_score": round(score, 1),
                        "sentiment_label": str(result.get('sentiment_label', 'neutral')),
                        "summary": str(result.get('summary', ''))[:200],
                        "item_count": len(news_items)
                    }
                except json.JSONDecodeError:
                    pass
            
            # If no JSON, try to parse sentiment from text
            if 'positive' in text.lower():
                score = 70
            elif 'negative' in text.lower():
                score = 30
            else:
                score = 50
            
            return {
                "news_score": score,
                "sentiment_label": "positive" if score > 60 else "negative" if score < 40 else "neutral",
                "summary": text[:150] if text else "Analysis unavailable",
                "item_count": len(news_items)
            }
            
        except Exception as e:
            logger.error(f"LLM news analysis error: {e}")
            return None
    
    def _keyword_fallback(self, news_items: List[Dict]) -> Dict:
        """Keyword-based sentiment fallback."""
        positive_words = [
            'surge', 'gain', 'rise', 'profit', 'growth', 'upgrade', 'buy',
            'bullish', 'strong', 'record', 'high', 'beat', 'exceed', 'positive',
            'expansion', 'acquisition', 'dividend', 'bonus'
        ]
        
        negative_words = [
            'fall', 'drop', 'decline', 'loss', 'downgrade', 'sell', 'bearish',
            'weak', 'low', 'miss', 'below', 'negative', 'fraud', 'probe',
            'investigation', 'penalty', 'fine', 'pledge', 'crash'
        ]
        
        pos_count = 0
        neg_count = 0
        
        for item in news_items:
            text = (item['title'] + ' ' + item.get('summary', '')).lower()
            
            pos_count += sum(1 for word in positive_words if word in text)
            neg_count += sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            score = 50
        else:
            score = 50 + (pos_count / total - 0.5) * 100
            score = max(0, min(100, score))
        
        label = "positive" if score > 60 else "negative" if score < 40 else "neutral"
        
        return {
            "news_score": round(score, 1),
            "sentiment_label": label,
            "summary": f"Keyword analysis: {pos_count} positive, {neg_count} negative signals",
            "item_count": len(news_items)
        }