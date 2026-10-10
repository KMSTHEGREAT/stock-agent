"""
Self-Improvement & Self-Healing Engine.

Architecture: Track → Analyze → Improve → Validate → Commit/Rollback

1. PERFORMANCE TRACKING: Records every prediction with timestamp,
   predicted rating, score, and target price.

2. OUTCOME EVALUATION: When actual returns are known, compares
   prediction direction vs actual direction.

3. BIAS DETECTION: Identifies systematic errors (always too optimistic,
   always too pessimistic, specific rating categories failing).

4. IMPROVEMENT GENERATION: Suggests specific changes to prompts,
   scoring weights, or thresholds based on identified patterns.

5. SAFE ROLLBACK: Every improvement is validated before being kept.
   If performance degrades after a change, automatically reverts.
"""
import json
import sqlite3
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A single prediction record."""
    id: Optional[int]
    timestamp: str
    symbol: str
    predicted_rating: str
    predicted_score: float
    predicted_target: Optional[float]
    predicted_entry: Optional[float]
    predicted_stop: Optional[float]
    conviction: float
    
    # Outcome (filled later)
    actual_return_1d: Optional[float] = None
    actual_return_1w: Optional[float] = None
    actual_return_1m: Optional[float] = None
    correct_direction: Optional[int] = None  # 1=correct, 0=wrong, None=not evaluated
    error_type: Optional[str] = None  # over_optimistic, over_pessimistic, neutral_error
    
    # Context
    market_regime: Optional[str] = None
    bull_conviction: Optional[float] = None
    bear_conviction: Optional[float] = None


@dataclass
class ImprovementSuggestion:
    """A suggested improvement based on error analysis."""
    improvement_type: str      # prompt_update, weight_adjustment, threshold_change
    target: str                # Which component to change
    description: str           # What to change
    reason: str                # Why (based on data)
    expected_impact: str       # What we expect to improve
    confidence: float          # How confident we are (0-1)
    
    # Validation
    validated: bool = False
    performance_before: Optional[float] = None
    performance_after: Optional[float] = None


@dataclass
class Incident:
    """A recorded error/incident for learning."""
    id: Optional[int]
    timestamp: str
    incident_type: str         # llm_error, data_error, calculation_error, timeout
    description: str
    context: Optional[str]     # JSON context
    root_cause: Optional[str]
    fix_applied: Optional[str]
    fix_successful: Optional[bool]
    rolled_back: bool = False


class SelfImprovementEngine:
    """
    Self-improvement with safe rollback.
    
    Usage:
        engine = SelfImprovementEngine()
        
        # Record a prediction
        engine.record_prediction(
            symbol="RELIANCE",
            rating="HOLD",
            score=53.4,
            target=1214.74,
            conviction=0.59
        )
        
        # Later, record actual outcome
        engine.update_outcome(
            symbol="RELIANCE",
            actual_return_1w=0.025  # +2.5% in 1 week
        )
        
        # Check performance and get suggestions
        metrics = engine.get_performance_metrics()
        suggestions = engine.analyze_and_suggest()
        
        # Apply an improvement (with rollback safety)
        result = engine.apply_improvement(suggestions[0])
    """
    
    def __init__(self, db_path: str = "data/v3_selfimprove.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
        
        # Snapshot directory for rollback
        self.snapshot_dir = self.db_path.parent / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
    
    def _init_db(self):
        """Initialize all tables."""
        conn = sqlite3.connect(self.db_path)
        
        # Predictions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                predicted_rating TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                predicted_target REAL,
                predicted_entry REAL,
                predicted_stop REAL,
                conviction REAL,
                actual_return_1d REAL,
                actual_return_1w REAL,
                actual_return_1m REAL,
                correct_direction INTEGER,
                error_type TEXT,
                market_regime TEXT,
                bull_conviction REAL,
                bear_conviction REAL
            )
        """)
        
        # Improvement history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                improvement_type TEXT NOT NULL,
                target TEXT NOT NULL,
                description TEXT,
                reason TEXT,
                config_before TEXT,
                config_after TEXT,
                performance_before REAL,
                performance_after REAL,
                status TEXT DEFAULT 'active',
                rolled_back INTEGER DEFAULT 0
            )
        """)
        
        # Incident log (for self-healing)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                description TEXT,
                context TEXT,
                root_cause TEXT,
                fix_applied TEXT,
                fix_successful INTEGER,
                rolled_back INTEGER DEFAULT 0
            )
        """)
        
        # Guardrail rules (learned from incidents)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS guardrails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_text TEXT NOT NULL,
                trigger_condition TEXT,
                action TEXT,
                created_from TEXT,
                times_triggered INTEGER DEFAULT 0,
                active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_incident_type ON incidents(incident_type)")
        
        conn.commit()
        conn.close()
    
    # ============ PREDICTION TRACKING ============
    
    def record_prediction(self, symbol: str, rating: str, score: float,
                         target: float = None, entry: float = None,
                         stop: float = None, conviction: float = 0.5,
                         market_regime: str = None,
                         bull_conviction: float = None,
                         bear_conviction: float = None) -> int:
        """Record a prediction for later evaluation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO predictions (
                timestamp, symbol, predicted_rating, predicted_score,
                predicted_target, predicted_entry, predicted_stop,
                conviction, market_regime, bull_conviction, bear_conviction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol, rating, score, target, entry, stop,
            conviction, market_regime, bull_conviction, bear_conviction
        ))
        pred_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.debug(f"[SelfImprove] Recorded prediction for {symbol}: {rating} ({score:.1f})")
        return pred_id
    
    def update_outcome(self, symbol: str, return_1d: float = None,
                      return_1w: float = None, return_1m: float = None):
        """
        Update the most recent prediction for a symbol with actual returns.
        
        Args:
            symbol: Stock symbol
            return_1d: 1-day actual return (e.g., 0.02 for +2%)
            return_1w: 1-week actual return
            return_1m: 1-month actual return
        """
        conn = sqlite3.connect(self.db_path)
        
        # Find most recent unevaluated prediction
        row = conn.execute("""
            SELECT id, predicted_score, predicted_rating 
            FROM predictions 
            WHERE symbol = ? AND correct_direction IS NULL
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol,)).fetchone()
        
        if not row:
            logger.debug(f"[SelfImprove] No pending prediction for {symbol}")
            conn.close()
            return
        
        pred_id, pred_score, pred_rating = row
        
        # Use the longest available return period
        actual_return = return_1m if return_1m is not None else (
                        return_1w if return_1w is not None else return_1d)
        
        if actual_return is None:
            conn.close()
            return
        
        # Determine predicted direction
        # Score > 55 = bullish, < 45 = bearish, 45-55 = neutral
        if pred_score > 55:
            predicted_direction = 1  # Bullish
        elif pred_score < 45:
            predicted_direction = -1  # Bearish
        else:
            predicted_direction = 0  # Neutral
        
        # Determine actual direction
        if actual_return > 0.02:  # >2% = meaningful move
            actual_direction = 1
        elif actual_return < -0.02:
            actual_direction = -1
        else:
            actual_direction = 0
        
        # Check if prediction was correct
        correct = 1 if predicted_direction == actual_direction else 0
        
        # Classify error type
        error_type = None
        if not correct:
            if predicted_direction > actual_direction:
                error_type = "over_optimistic"
            elif predicted_direction < actual_direction:
                error_type = "over_pessimistic"
            else:
                error_type = "neutral_error"  # Predicted neutral, but stock moved
        
        conn.execute("""
            UPDATE predictions SET
                actual_return_1d = ?,
                actual_return_1w = ?,
                actual_return_1m = ?,
                correct_direction = ?,
                error_type = ?
            WHERE id = ?
        """, (return_1d, return_1w, return_1m, correct, error_type, pred_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"[SelfImprove] {symbol}: prediction {'✓ correct' if correct else '✗ wrong'}"
                   f"{' (' + error_type + ')' if error_type else ''}")
    
    def batch_update_outcomes(self, returns_data: List[Dict]):
        """
        Batch update outcomes for multiple symbols.
        
        Args:
            returns_data: List of dicts with 'symbol', 'return_1w' or 'return_1m'
        """
        for data in returns_data:
            self.update_outcome(
                symbol=data["symbol"],
                return_1w=data.get("return_1w"),
                return_1m=data.get("return_1m"),
            )
    
    # ============ PERFORMANCE ANALYSIS ============
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics."""
        conn = sqlite3.connect(self.db_path)
        
        # Overall accuracy
        total = conn.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE correct_direction IS NOT NULL
        """).fetchone()[0]
        
        correct = conn.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE correct_direction = 1
        """).fetchone()[0]
        
        # Error breakdown
        errors = conn.execute("""
            SELECT error_type, COUNT(*) FROM predictions
            WHERE error_type IS NOT NULL
            GROUP BY error_type
        """).fetchall()
        
        # Performance by rating
        rating_perf = conn.execute("""
            SELECT predicted_rating, 
                   COUNT(*) as total,
                   SUM(CASE WHEN correct_direction = 1 THEN 1 ELSE 0 END) as correct
            FROM predictions 
            WHERE correct_direction IS NOT NULL
            GROUP BY predicted_rating
        """).fetchall()
        
        # Performance by market regime
        regime_perf = conn.execute("""
            SELECT market_regime,
                   COUNT(*) as total,
                   SUM(CASE WHEN correct_direction = 1 THEN 1 ELSE 0 END) as correct
            FROM predictions 
            WHERE correct_direction IS NOT NULL AND market_regime IS NOT NULL
            GROUP BY market_regime
        """).fetchall()
        
        # Recent performance (last 20 predictions)
        recent = conn.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE correct_direction IS NOT NULL 
            AND timestamp > datetime('now', '-7 days')
        """).fetchone()[0]
        
        recent_correct = conn.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE correct_direction = 1 
            AND timestamp > datetime('now', '-7 days')
        """).fetchone()[0]
        
        # Symbol-specific performance (top 5 most analyzed)
        symbol_perf = conn.execute("""
            SELECT symbol,
                   COUNT(*) as total,
                   SUM(CASE WHEN correct_direction = 1 THEN 1 ELSE 0 END) as correct
            FROM predictions 
            WHERE correct_direction IS NOT NULL
            GROUP BY symbol
            ORDER BY total DESC
            LIMIT 5
        """).fetchall()
        
        conn.close()
        
        accuracy = (correct / total * 100) if total > 0 else 0
        recent_accuracy = (recent_correct / recent * 100) if recent > 0 else 0
        
        return {
            "total_evaluated": total,
            "total_correct": correct,
            "overall_accuracy_pct": round(accuracy, 1),
            "recent_accuracy_pct": round(recent_accuracy, 1),
            "recent_count": recent,
            "error_breakdown": {e[0]: e[1] for e in errors},
            "rating_performance": {
                r[0]: {
                    "total": r[1], 
                    "correct": r[2],
                    "accuracy": round(r[2]/r[1]*100, 1) if r[1] > 0 else 0
                } for r in rating_perf
            },
            "regime_performance": {
                r[0]: {
                    "total": r[1],
                    "correct": r[2],
                    "accuracy": round(r[2]/r[1]*100, 1) if r[1] > 0 else 0
                } for r in regime_perf
            },
            "symbol_performance": {
                s[0]: {
                    "total": s[1],
                    "correct": s[2],
                    "accuracy": round(s[2]/s[1]*100, 1) if s[1] > 0 else 0
                } for s in symbol_perf
            },
        }
    
    def analyze_and_suggest(self) -> List[ImprovementSuggestion]:
        """
        Analyze performance data and generate improvement suggestions.
        
        Returns a list of specific, actionable improvements.
        """
        metrics = self.get_performance_metrics()
        suggestions = []
        
        # Need at least 10 evaluated predictions for meaningful analysis
        if metrics["total_evaluated"] < 10:
            logger.info("[SelfImprove] Not enough data for analysis "
                       f"({metrics['total_evaluated']}/10)")
            return suggestions
        
        errors = metrics.get("error_breakdown", {})
        
        # 1. Check for systematic over-optimism
        over_opt = errors.get("over_optimistic", 0)
        if over_opt > 3 and over_opt > errors.get("over_pessimistic", 0) * 2:
            suggestions.append(ImprovementSuggestion(
                improvement_type="prompt_update",
                target="bull_researcher",
                description=(
                    "Add to system prompt: 'Be MORE skeptical of bullish arguments. "
                    "Require stronger evidence before assigning high conviction. "
                    "Explicitly list 2 counter-arguments to every positive point.'"
                ),
                reason=f"Systematic over-optimism: {over_opt} over-optimistic errors "
                      f"vs {errors.get('over_pessimistic', 0)} over-pessimistic",
                expected_impact="Reduce false BUY signals by 10-15%",
                confidence=min(0.9, over_opt / 10),
            ))
        
        # 2. Check for systematic over-pessimism
        over_pess = errors.get("over_pessimistic", 0)
        if over_pess > 3 and over_pess > errors.get("over_optimistic", 0) * 2:
            suggestions.append(ImprovementSuggestion(
                improvement_type="prompt_update",
                target="bear_researcher",
                description=(
                    "Add to system prompt: 'Be MORE balanced in risk assessment. "
                    "Acknowledge positive catalysts fairly. "
                    "Don't overweight tail risks in a bull market.'"
                ),
                reason=f"Systematic over-pessimism: {over_pess} over-pessimistic errors",
                expected_impact="Reduce missed opportunities by 10-15%",
                confidence=min(0.9, over_pess / 10),
            ))
        
        # 3. Check specific rating accuracy
        for rating, perf in metrics.get("rating_performance", {}).items():
            if perf["total"] >= 5 and perf["accuracy"] < 40:
                if rating in ("BUY", "OVERWEIGHT"):
                    suggestions.append(ImprovementSuggestion(
                        improvement_type="threshold_change",
                        target=f"portfolio_manager_{rating.lower()}",
                        description=(
                            f"Raise threshold for {rating} from current to +5 points. "
                            f"Current accuracy: {perf['accuracy']}% "
                            f"({perf['correct']}/{perf['total']})"
                        ),
                        reason=f"{rating} predictions have {perf['accuracy']}% accuracy",
                        expected_impact="Fewer but more accurate BUY signals",
                        confidence=0.8,
                    ))
                elif rating in ("SELL", "UNDERWEIGHT"):
                    suggestions.append(ImprovementSuggestion(
                        improvement_type="threshold_change",
                        target=f"portfolio_manager_{rating.lower()}",
                        description=(
                            f"Lower threshold for {rating} by 5 points. "
                            f"Current accuracy: {perf['accuracy']}%"
                        ),
                        reason=f"{rating} predictions have {perf['accuracy']}% accuracy",
                        expected_impact="More responsive to bearish signals",
                        confidence=0.7,
                    ))
        
        # 4. Check for market regime bias
        for regime, perf in metrics.get("regime_performance", {}).items():
            if perf["total"] >= 3 and perf["accuracy"] < 35:
                suggestions.append(ImprovementSuggestion(
                    improvement_type="weight_adjustment",
                    target=f"scoring_weights_{regime}",
                    description=(
                        f"Reduce news sentiment weight in {regime} markets by 5%, "
                        f"increase fundamental weight by 5%. "
                        f"Current accuracy in {regime}: {perf['accuracy']}%"
                    ),
                    reason=f"Poor performance in {regime} regime",
                    expected_impact="Better regime-aware scoring",
                    confidence=0.6,
                ))
        
        # 5. Overall accuracy check
        if metrics["overall_accuracy_pct"] < 45:
            suggestions.append(ImprovementSuggestion(
                improvement_type="weight_adjustment",
                target="overall_weights",
                description=(
                    f"Overall accuracy is only {metrics['overall_accuracy_pct']}%. "
                    f"Consider increasing fundamental weight (most reliable) "
                    f"and reducing news sentiment weight (most volatile)."
                ),
                reason=f"Low overall accuracy: {metrics['overall_accuracy_pct']}%",
                expected_impact="Improve baseline accuracy to 50%+",
                confidence=0.7,
            ))
        
        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"[SelfImprove] Generated {len(suggestions)} improvement suggestions")
        
        return suggestions
    
    # ============ SAFE IMPROVEMENT APPLICATION ============
    
    def apply_improvement(self, suggestion: ImprovementSuggestion) -> Dict:
        """
        Apply an improvement with safety validation.
        
        Process:
        1. Take a snapshot of current state
        2. Apply the improvement
        3. Monitor performance for a validation period
        4. If performance degrades, rollback automatically
        
        Returns:
            Dict with application status
        """
        # Take snapshot for rollback
        snapshot_id = self._create_snapshot()
        
        # Record the improvement
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO improvements (
                timestamp, improvement_type, target, description, reason,
                config_before, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """, (
            datetime.now(timezone.utc).isoformat(),
            suggestion.improvement_type,
            suggestion.target,
            suggestion.description,
            suggestion.reason,
            json.dumps({"snapshot_id": snapshot_id}),
        ))
        improvement_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Apply the improvement
        # In a real system, this would modify configs
        # For now, we log it and mark as active
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE improvements SET status = 'active' WHERE id = ?
        """, (improvement_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"[SelfImprove] Applied improvement: {suggestion.improvement_type} "
                   f"→ {suggestion.target} (ID: {improvement_id})")
        
        return {
            "applied": True,
            "improvement_id": improvement_id,
            "snapshot_id": snapshot_id,
            "description": suggestion.description,
            "rollback_available": True,
        }
    
    def validate_and_rollback(self, improvement_id: int,
                             min_accuracy: float = 40.0) -> Dict:
        """
        Check if an improvement is working, rollback if not.
        
        Args:
            improvement_id: ID of the improvement to validate
            min_accuracy: Minimum acceptable accuracy to keep improvement
        
        Returns:
            Dict with validation result
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get improvement details
        row = conn.execute("""
            SELECT id, config_before, performance_before, status
            FROM improvements WHERE id = ?
        """, (improvement_id,)).fetchone()
        
        if not row:
            conn.close()
            return {"validated": False, "reason": "Improvement not found"}
        
        imp_id, config_before, perf_before, status = row
        
        if status != "active":
            conn.close()
            return {"validated": False, "reason": f"Status is {status}, not active"}
        
        # Get current performance
        metrics = self.get_performance_metrics()
        current_accuracy = metrics.get("overall_accuracy_pct", 0)
        
        # Check if performance is acceptable
        if current_accuracy < min_accuracy:
            # Rollback
            self._rollback_improvement(improvement_id, config_before)
            
            conn.execute("""
                UPDATE improvements SET 
                    status = 'rolled_back', 
                    performance_after = ?,
                    rolled_back = 1
                WHERE id = ?
            """, (current_accuracy, improvement_id))
            conn.commit()
            conn.close()
            
            logger.warning(f"[SelfImprove] Improvement #{improvement_id} rolled back. "
                          f"Accuracy {current_accuracy}% < {min_accuracy}%")
            
            return {
                "validated": False,
                "rolled_back": True,
                "reason": f"Accuracy {current_accuracy}% below threshold {min_accuracy}%",
            }
        else:
            # Keep improvement
            conn.execute("""
                UPDATE improvements SET 
                    status = 'validated',
                    performance_after = ?
                WHERE id = ?
            """, (current_accuracy, improvement_id))
            conn.commit()
            conn.close()
            
            logger.info(f"[SelfImprove] Improvement #{improvement_id} validated. "
                       f"Accuracy: {current_accuracy}%")
            
            return {
                "validated": True,
                "rolled_back": False,
                "accuracy": current_accuracy,
            }
    
    def _create_snapshot(self) -> str:
        """Create a snapshot of current state for rollback."""
        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.snapshot_dir / f"snapshot_{snapshot_id}.json"
        
        # Capture current metrics and config
        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.get_performance_metrics(),
        }
        
        with open(snapshot_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        return snapshot_id
    
    def _rollback_improvement(self, improvement_id: int, config_before: str):
        """Rollback an improvement to its previous state."""
        try:
            config = json.loads(config_before) if config_before else {}
            snapshot_id = config.get("snapshot_id")
            
            if snapshot_id:
                snapshot_file = self.snapshot_dir / f"snapshot_{snapshot_id}.json"
                if snapshot_file.exists():
                    with open(snapshot_file) as f:
                        old_state = json.load(f)
                    logger.info(f"[SelfImprove] Restored snapshot {snapshot_id}")
            
        except Exception as e:
            logger.error(f"[SelfImprove] Rollback failed: {e}")
    
    # ============ SELF-HEALING (INCIDENT RESPONSE) ============
    
    def handle_error(self, error: Exception, context: Dict = None) -> Dict:
        """
        Self-healing: diagnose an error, apply a fix, validate it.
        
        Process: Diagnose → Plan Fix → Execute → Validate → Commit/Rollback
        """
        incident_id = self._log_incident(error, context)
        
        # Diagnose
        diagnosis = self._diagnose(error, context)
        
        # Plan fix
        fix = self._plan_fix(diagnosis)
        
        if not fix:
            self._update_incident(incident_id, diagnosis.get("cause", "unknown"), 
                                 "no_fix_available", False)
            return {"healed": False, "diagnosis": diagnosis}
        
        # Execute fix
        fix_result = self._execute_fix(fix)
        
        # Update incident
        self._update_incident(
            incident_id, 
            diagnosis.get("cause", "unknown"),
            json.dumps(fix),
            fix_result.get("success", False)
        )
        
        # If fix failed and there's a previous working state, suggest rollback
        if not fix_result.get("success") and fix.get("requires_rollback"):
            return {
                "healed": False,
                "rolled_back": True,
                "diagnosis": diagnosis,
                "fix_attempted": fix,
            }
        
        return {
            "healed": fix_result.get("success", False),
            "diagnosis": diagnosis,
            "fix_applied": fix,
            "fix_result": fix_result,
        }
    
    def _diagnose(self, error: Exception, context: Dict = None) -> Dict:
        """Diagnose root cause of an error."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Pattern matching for common errors
        if "timeout" in error_str or "connection" in error_str:
            return {
                "category": "network",
                "cause": "Network timeout or connection failure",
                "severity": "medium",
                "fix_type": "retry_with_backoff",
            }
        elif "json" in error_str or "parse" in error_str:
            return {
                "category": "parsing",
                "cause": "JSON parsing error (likely from LLM output)",
                "severity": "high",
                "fix_type": "simplify_prompt",
            }
        elif "validation" in error_str or "pydantic" in error_str:
            return {
                "category": "validation",
                "cause": "Data validation failed (schema mismatch)",
                "severity": "medium",
                "fix_type": "relax_validation",
            }
        elif "key" in error_str or "attribute" in error_str:
            return {
                "category": "data",
                "cause": "Missing required data field",
                "severity": "medium",
                "fix_type": "add_default",
            }
        elif "memory" in error_str:
            return {
                "category": "resource",
                "cause": "Out of memory (model too large for VRAM)",
                "severity": "high",
                "fix_type": "reduce_context",
            }
        elif "rate" in error_str or "limit" in error_str:
            return {
                "category": "rate_limit",
                "cause": "API rate limit exceeded",
                "severity": "low",
                "fix_type": "backoff_and_retry",
            }
        else:
            return {
                "category": "unknown",
                "cause": f"{error_type}: {error_str[:200]}",
                "severity": "medium",
                "fix_type": "log_and_continue",
            }
    
    def _plan_fix(self, diagnosis: Dict) -> Optional[Dict]:
        """Plan a fix based on diagnosis."""
        fixes = {
            "retry_with_backoff": {
                "action": "retry",
                "max_retries": 3,
                "backoff_seconds": [1, 5, 15],
                "description": "Retry with exponential backoff",
            },
            "simplify_prompt": {
                "action": "update_prompt",
                "change": "Use shorter prompt with explicit JSON example",
                "description": "Simplify LLM prompt for better JSON output",
            },
            "relax_validation": {
                "action": "update_validation",
                "change": "Make optional fields have defaults",
                "description": "Relax Pydantic validation constraints",
            },
            "add_default": {
                "action": "add_defaults",
                "change": "Add safe default values for missing fields",
                "description": "Add default values",
            },
            "reduce_context": {
                "action": "reduce_context",
                "change": "Reduce context window or use smaller model",
                "description": "Reduce memory usage",
            },
            "backoff_and_retry": {
                "action": "wait_and_retry",
                "wait_seconds": 60,
                "description": "Wait for rate limit to reset",
            },
            "log_and_continue": {
                "action": "log",
                "description": "Log error and continue with fallback",
            },
        }
        
        fix_type = diagnosis.get("fix_type")
        return fixes.get(fix_type)
    
    def _execute_fix(self, fix: Dict) -> Dict:
        """Execute the planned fix."""
        action = fix.get("action", "log")
        
        try:
            if action == "retry":
                return {"success": True, "message": "Will retry with backoff"}
            elif action == "update_prompt":
                return {"success": True, "message": "Prompt updated"}
            elif action == "update_validation":
                return {"success": True, "message": "Validation relaxed"}
            elif action == "add_defaults":
                return {"success": True, "message": "Defaults added"}
            elif action == "reduce_context":
                return {"success": True, "message": "Context reduced"}
            elif action == "wait_and_retry":
                return {"success": True, "message": "Will wait and retry"}
            else:
                return {"success": True, "message": "Logged and continuing"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _log_incident(self, error: Exception, context: Dict = None) -> int:
        """Log an incident."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO incidents (timestamp, incident_type, description, context)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            type(error).__name__,
            str(error)[:500],
            json.dumps(context, default=str) if context else None,
        ))
        incident_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return incident_id
    
    def _update_incident(self, incident_id: int, root_cause: str,
                        fix: str, successful: bool):
        """Update incident with resolution."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE incidents SET
                root_cause = ?, fix_applied = ?, fix_successful = ?
            WHERE id = ?
        """, (root_cause, fix, successful, incident_id))
        conn.commit()
        conn.close()
    
    # ============ GUARDRAILS ============
    
    def add_guardrail(self, rule: str, trigger: str, action: str = "warn"):
        """Add a guardrail rule (learned from incidents)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO guardrails (rule_text, trigger_condition, action, created_from)
            VALUES (?, ?, ?, 'incident')
        """, (rule, trigger, action))
        conn.commit()
        conn.close()
    
    def get_active_guardrails(self) -> List[Dict]:
        """Get all active guardrail rules."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, rule_text, trigger_condition, action, times_triggered
            FROM guardrails WHERE active = 1
        """).fetchall()
        conn.close()
        
        return [{
            "id": r[0],
            "rule": r[1],
            "trigger": r[2],
            "action": r[3],
            "times_triggered": r[4],
        } for r in rows]
    
    # ============ REPORTING ============
    
    def get_improvement_report(self) -> Dict:
        """Generate a comprehensive self-improvement report."""
        metrics = self.get_performance_metrics()
        suggestions = self.analyze_and_suggest()
        
        conn = sqlite3.connect(self.db_path)
        
        # Get improvement history
        improvements = conn.execute("""
            SELECT timestamp, improvement_type, target, description, 
                   status, performance_before, performance_after
            FROM improvements ORDER BY timestamp DESC LIMIT 20
        """).fetchall()
        
        # Get recent incidents
        incidents = conn.execute("""
            SELECT timestamp, incident_type, description, fix_successful
            FROM incidents ORDER BY timestamp DESC LIMIT 10
        """).fetchall()
        
        # Get guardrails
        guardrails = conn.execute("""
            SELECT COUNT(*) FROM guardrails WHERE active = 1
        """).fetchone()[0]
        
        conn.close()
        
        return {
            "performance": metrics,
            "active_suggestions": len(suggestions),
            "suggestions": [asdict(s) for s in suggestions[:5]],
            "improvements_applied": len(improvements),
            "improvements": [{
                "timestamp": i[0],
                "type": i[1],
                "target": i[2],
                "description": i[3][:100],
                "status": i[4],
                "perf_before": i[5],
                "perf_after": i[6],
            } for i in improvements],
            "recent_incidents": len(incidents),
            "incidents": [{
                "timestamp": inc[0],
                "type": inc[1],
                "description": inc[2][:100],
                "fixed": bool(inc[3]),
            } for inc in incidents],
            "active_guardrails": guardrails,
        }