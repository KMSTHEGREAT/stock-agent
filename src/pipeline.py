"""
Complete analysis pipeline integrating all components.
This is the main orchestrator for stock analysis.
"""
import json
import sqlite3
from typing import Optional, Dict, List
from datetime import datetime, timezone

from src.data.price_fetcher import PriceFetcher
from src.data.symbol_mapper import SymbolMapper
from src.data.cache import DataCache
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.fundamental import FundamentalAnalyzer
from src.analysis.market_context import MarketContextAnalyzer
from src.analysis.technical_scorer import TechnicalScorer
from src.analysis.fundamental_scorer import FundamentalScorer
from src.news.aggregator import IndianNewsAggregator
from src.news.sentiment import NewsSentimentScorer
from src.scoring.engine import ScoringEngine
from src.llm.synthesizer import LLMSynthesizer
from src.recommendations.generator import RecommendationGenerator
from src.schemas.stock_analysis import FundamentalMetrics
from src.utils.logger import setup_logger
from config.settings import DATABASE

logger = setup_logger(__name__)


class StockAnalysisPipeline:
    """Main pipeline for analyzing stocks."""
    
    def __init__(self):
        # Data layer
        self.cache = DataCache()
        self.symbol_mapper = SymbolMapper()
        self.price_fetcher = PriceFetcher(self.cache, self.symbol_mapper)
        
        # Analysis layer
        self.technical_analyzer = TechnicalAnalyzer(self.price_fetcher)
        self.fundamental_analyzer = FundamentalAnalyzer(self.cache, self.price_fetcher)
        self.market_context_analyzer = MarketContextAnalyzer(self.cache)
        
        # Scoring layer
        self.technical_scorer = TechnicalScorer()
        self.fundamental_scorer = FundamentalScorer()
        self.scoring_engine = ScoringEngine()
        
        # News layer
        self.news_aggregator = IndianNewsAggregator(self.cache)
        self.news_scorer = NewsSentimentScorer(self.news_aggregator)
        
        # LLM layer
        self.llm_synthesizer = LLMSynthesizer()
        
        # Recommendation layer
        self.recommendation_generator = RecommendationGenerator()
        
        # Database
        self.db_path = DATABASE["path"]
        self._init_db()
    
    def analyze_stock(self, symbol: str, include_news: bool = True,
                 include_llm: bool = True) -> Optional[Dict]:
        """Analyze a stock. Accepts NSE symbols or BSE codes."""
        
        # Resolve to NSE if BSE code provided
        try:
            # First validate
            symbol_info = self.symbol_mapper.validate_symbol(symbol.upper())
            
            # If it's a BSE code, resolve to NSE
            if symbol_info.get("bse_code") == symbol.upper() and not symbol_info.get("nse_symbol"):
                # This is a BSE-only entry, try to resolve
                nse_symbol = self.symbol_mapper.resolve_to_nse(symbol.upper())
                symbol = nse_symbol
                symbol_info = self.symbol_mapper.validate_symbol(nse_symbol)
            elif symbol_info.get("nse_symbol"):
                symbol = symbol_info["nse_symbol"]
            
            company_name = symbol_info.get("company_name") or symbol
            start_time = datetime.now(timezone.utc)
            
        except ValueError as e:
            logger.error(f"Invalid symbol: {e}")
            return None
        
        result = {
            "symbol": symbol,
            "company_name": company_name,
            "analysis_timestamp": start_time.isoformat(),
            "currency": "INR",
        }
        
        try:
            # Step 1: Market Context
            print(f"  [1/6] Market context...")
            market_context = self.market_context_analyzer.get_market_context()
            macro_score = self.market_context_analyzer.get_macro_score(market_context)
            result["market_context"] = {
                "regime": market_context.market_regime,
                "nifty_trend": market_context.nifty_trend.value,
                "sensex_trend": market_context.sensex_trend.value,
                "vix": market_context.vix_level,
            }
            result["macro_score"] = macro_score
            
            # Step 2: Technical Analysis
            print(f"  [2/6] Technical analysis...")
            technical_metrics = self.technical_analyzer.analyze(symbol, period="1y")
            
            if technical_metrics is None:
                logger.error(f"Technical analysis failed for {symbol}")
                return None
            
            technical_scores = self.technical_scorer.score(technical_metrics)
            result["technical_metrics"] = technical_metrics.model_dump()
            result["technical_scores"] = technical_scores
            
            # Step 3: Fundamental Analysis
            print(f"  [3/6] Fundamental analysis...")
            fundamental_metrics = self.fundamental_analyzer.analyze(symbol)
            
            if fundamental_metrics and isinstance(fundamental_metrics, FundamentalMetrics):
                fundamental_scores = self.fundamental_scorer.score(fundamental_metrics)
                result["fundamental_metrics"] = fundamental_metrics.model_dump()
            else:
                fundamental_metrics = FundamentalMetrics(symbol=symbol)
                fundamental_scores = {"total": 50}
                result["fundamental_metrics"] = None
            
            result["fundamental_scores"] = fundamental_scores
            
            # Step 4: News Analysis
            if include_news:
                print(f"  [4/6] News analysis...")
                try:
                    news_result = self.news_scorer.score_news_sentiment(symbol, company_name)
                    result["news_analysis"] = news_result
                    news_score = news_result.get("news_score", 50)
                except Exception as e:
                    logger.warning(f"News analysis failed: {e}")
                    result["news_analysis"] = None
                    news_score = 50
            else:
                result["news_analysis"] = None
                news_score = 50
            
            # Step 5: Overall Score
            print(f"  [5/6] Computing overall score...")
            scores = self.scoring_engine.compute_score(
                technical_metrics=technical_metrics,
                fundamental_metrics=fundamental_metrics,
                news_score=news_score,
                macro_score=macro_score
            )
            
            result["scores"] = scores
            result["overall_score"] = scores["overall_score"]
            result["recommendation"] = scores["recommendation"].value
            
            # Step 6: Recommendations
            print(f"  [6/6] Generating recommendations...")
            recommendation = self.recommendation_generator.generate(
                symbol=symbol,
                company_name=company_name,
                technical_metrics=technical_metrics,
                recommendation=scores["recommendation"],
                overall_score=scores["overall_score"]
            )
            
            result["recommendation_details"] = recommendation
            
            # Step 7: LLM Analysis
            if include_llm:
                print(f"  [7/7] LLM synthesis...")
                news_context = ""
                if result.get("news_analysis"):
                    news_context = result["news_analysis"].get("summary", "")
                
                try:
                    llm_result = self.llm_synthesizer.synthesize_analysis(
                        symbol=symbol,
                        company_name=company_name,
                        technical_metrics=technical_metrics,
                        technical_scores=technical_scores,
                        fundamental_metrics=fundamental_metrics,
                        fundamental_scores=fundamental_scores,
                        news_context=news_context
                    )
                    result["llm_analysis"] = llm_result
                except Exception as e:
                    logger.warning(f"LLM synthesis failed: {e}")
                    result["llm_analysis"] = None
            
            # Processing time
            result["processing_time_seconds"] = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Save to database
            self._save_result(result)
            
            logger.info(f"Analysis complete: {symbol} = {result['overall_score']}/100")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_watchlist(self, symbols: List[str], 
                         include_news: bool = False) -> List[Dict]:
        """Analyze multiple stocks."""
        results = []
        
        print(f"\nAnalyzing {len(symbols)} stocks...")
        print("=" * 60)
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {symbol}")
            print("-" * 30)
            
            result = self.analyze_stock(symbol, include_news=include_news)
            if result:
                results.append(result)
        
        # Sort by score
        results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
        return results
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT,
                analysis_timestamp TEXT,
                overall_score REAL,
                technical_score REAL,
                fundamental_score REAL,
                news_score REAL,
                macro_score REAL,
                recommendation TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                current_price REAL,
                analysis_json TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                company_name TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_results_symbol 
            ON analysis_results(symbol)
        """)
        
        conn.commit()
        conn.close()
    
    def _save_result(self, result: Dict):
        """Save analysis result to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            rec = result.get("recommendation_details", {})
            scores = result.get("scores", {})
            
            conn.execute("""
                INSERT INTO analysis_results (
                    symbol, company_name, analysis_timestamp,
                    overall_score, technical_score, fundamental_score,
                    news_score, macro_score, recommendation,
                    entry_price, target_price, stop_loss, current_price,
                    analysis_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["symbol"],
                result["company_name"],
                result["analysis_timestamp"],
                result.get("overall_score", 0),
                scores.get("technical_score", 0),
                scores.get("fundamental_score", 0),
                scores.get("news_score", 0),
                scores.get("macro_score", 0),
                result.get("recommendation", ""),
                rec.get("entry_price"),
                rec.get("target_price"),
                rec.get("stop_loss"),
                rec.get("current_price"),
                json.dumps(result, default=str)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
    
    def add_to_watchlist(self, symbol: str, company_name: str = None):
        """Add symbol to watchlist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (symbol, company_name) VALUES (?, ?)",
            (symbol, company_name)
        )
        conn.commit()
        conn.close()
    
    def remove_from_watchlist(self, symbol: str):
        """Remove symbol from watchlist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()
    
    def get_watchlist(self) -> List[Dict]:
        """Get watchlist symbols."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT symbol, company_name FROM watchlist ORDER BY symbol"
        ).fetchall()
        conn.close()
        return [{"symbol": r[0], "company_name": r[1]} for r in rows]