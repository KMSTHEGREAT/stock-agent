"""
LLM synthesis layer for Qwen 3 4B.
Qwen 3 puts thinking in message.thinking and answer in message.content.
When content is empty, extract from thinking. Parse natural language as fallback.
"""
import json
import re
import ollama
from typing import Optional, Dict, List

from v1.schemas.stock_analysis import (
    TechnicalMetrics, FundamentalMetrics, MarketContext
)
from v1.utils.logger import setup_logger
from config.settings import LLM

logger = setup_logger(__name__)


class LLMSynthesizer:
    """Uses Qwen 3 4B to synthesize qualitative analysis."""
    
    def __init__(self, model: str = None):
        self.model = model or LLM["model"]
    
    def _call_llm(self, prompt: str, system: str = None,
                  temperature: float = 0.3,
                  num_predict: int = 800) -> Optional[str]:
        """Call LLM and return content (or extract from thinking)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": num_predict,
                }
            )
            
            msg = response.message
            content = msg.content or ""
            thinking = msg.thinking or ""
            
            # Return content if available
            if content.strip():
                return content
            
            # Otherwise, use thinking
            if thinking.strip():
                logger.info(f"Content empty, using thinking ({len(thinking)} chars)")
                return thinking
            
            return ""
            
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return None
    
    def _find_json(self, text: str) -> Optional[str]:
        """Find JSON object in text."""
        if not text:
            return None
        
        # Find { } boundaries with proper nesting
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"' and not escape:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except:
                            # Try fixing common issues
                            fixed = re.sub(r',\s*}', '}', candidate)
                            fixed = re.sub(r',\s*]', ']', fixed)
                            try:
                                json.loads(fixed)
                                return fixed
                            except:
                                continue
        
        return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON dict from text."""
        if not text:
            return None
        
        # Clean text
        cleaned = text.strip()
        
        # Remove markdown code blocks
        code_match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
        if code_match:
            cleaned = code_match.group(1).strip()
        
        # Try direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        
        # Find JSON in text
        json_str = self._find_json(cleaned)
        if json_str:
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
            except:
                pass
        
        return None
    
    def _clean_analysis_text(self, text: str) -> str:
        """Remove meta-commentary and prompt echoes from analysis text."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # Skip patterns that indicate meta-commentary or prompt echoing
        skip_patterns = [
            r'^we are given',
            r'^the user',
            r'^let me',
            r'^i need to',
            r'^first,',
            r'^okay',
            r'^hmm',
            r'^technical:',
            r'^fundamental:',
            r'^news:',
            r'^respond with',
            r'^json',
            r'^\{',
            r'^\}',
            r'^score \d+',
            r'^price ₹',
            r'^rsi \d+',
            r'^p/e',
            r'^roce',
            r'^promoter',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that match meta patterns
            should_skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    should_skip = True
                    break
            
            if not should_skip:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


    def _parse_natural_language(self, text: str, symbol: str) -> Optional[Dict]:
        """Parse natural language analysis with cleanup."""
        if not text:
            return None
        
        logger.info(f"Parsing natural language output for {symbol}")
        
        # Clean the text first
        cleaned = self._clean_analysis_text(text)
        
        if not cleaned.strip():
            # If cleaning removed everything, use original
            cleaned = text
        
        result = {
            "analysis_summary": "",
            "key_risks": [],
            "key_catalysts": [],
            "confidence_level": 0.5,
            "risk_flags": {}
        }
        
        # Extract summary from cleaned text
        sentences = re.split(r'[.!?]\s*', cleaned)
        meaningful = [s.strip() for s in sentences 
                    if s.strip() and len(s.strip()) > 15 
                    and not s.strip().startswith(('We', 'The user', 'Let'))]
        
        if meaningful:
            result["analysis_summary"] = '. '.join(meaningful[:3]) + '.'
        elif sentences:
            # Fallback to first meaningful sentence from original
            for s in sentences:
                if len(s.strip()) > 15:
                    result["analysis_summary"] = s.strip() + '.'
                    break
        
        # Extract risks (look in both cleaned and original)
        for source_text in [cleaned, text]:
            risk_patterns = [
                r'risks?\s*[:\-]\s*(.*?)(?=\n|catalyst|positive|upside|$)',
                r'concerns?\s*[:\-]\s*(.*?)(?=\n|catalyst|positive|upside|$)',
            ]
            
            for pattern in risk_patterns:
                match = re.search(pattern, source_text, re.IGNORECASE | re.DOTALL)
                if match:
                    risk_text = match.group(1).strip()
                    # Split by common list markers
                    risks = re.split(r'(?:^|\n)\s*(?:[a-z]\)|\d+\.|•|-)\s*', risk_text)
                    risks = [r.strip().rstrip('.') for r in risks 
                            if r.strip() and len(r.strip()) > 10 
                            and not r.strip().startswith(('N/A', 'which'))]
                    if risks:
                        result["key_risks"] = risks[:5]
                        break
            if result["key_risks"]:
                break
        
        # Extract catalysts
        for source_text in [cleaned, text]:
            catalyst_patterns = [
                r'catalysts?\s*[:\-]\s*(.*?)(?=\n|risk|concern|downside|$)',
                r'positives?\s*[:\-]\s*(.*?)(?=\n|risk|concern|downside|$)',
            ]
            
            for pattern in catalyst_patterns:
                match = re.search(pattern, source_text, re.IGNORECASE | re.DOTALL)
                if match:
                    catalyst_text = match.group(1).strip()
                    catalysts = re.split(r'(?:^|\n)\s*(?:[a-z]\)|\d+\.|•|-)\s*', catalyst_text)
                    catalysts = [c.strip().rstrip('.') for c in catalysts 
                                if c.strip() and len(c.strip()) > 10 
                                and 'note' not in c.lower()]
                    if catalysts:
                        result["key_catalysts"] = catalysts[:5]
                        break
            if result["key_catalysts"]:
                break
        
        # Confidence based on sentiment
        if re.search(r'(strong|high|very)\s+(positive|confident|bullish)', text, re.I):
            result["confidence_level"] = 0.8
        elif re.search(r'moderate', text, re.I):
            result["confidence_level"] = 0.6
        elif re.search(r'(low|weak|uncertain)', text, re.I):
            result["confidence_level"] = 0.4
        
        # Only return if we have at least a summary
        if result["analysis_summary"] and len(result["analysis_summary"]) > 20:
            return result
        
        return None
    
    def synthesize_analysis(
        self,
        symbol: str,
        company_name: str,
        technical_metrics: Optional[TechnicalMetrics],
        technical_scores: Dict,
        fundamental_metrics: Optional[FundamentalMetrics],
        fundamental_scores: Dict,
        news_context: str = "",
        market_context: Optional[MarketContext] = None
    ) -> Optional[Dict]:
        """Generate qualitative analysis using LLM."""
        logger.info(f"Starting LLM synthesis for {symbol}")
        
        # Build prompt
        prompt = self._build_prompt(
            symbol, company_name,
            technical_metrics, technical_scores,
            fundamental_metrics, fundamental_scores,
            news_context
        )
        
        # System prompt that strongly encourages JSON
        system = """You are a stock analyst. You MUST respond with valid JSON only.

Example response format:
{
  "analysis_summary": "Brief 2-3 sentence analysis",
  "key_risks": ["risk 1", "risk 2"],
  "key_catalysts": ["catalyst 1", "catalyst 2"],
  "confidence_level": 0.7
}

Do NOT include any text before or after the JSON. Output ONLY the JSON object."""
        
        # Call LLM
        text = self._call_llm(
            prompt=prompt,
            system=system,
            temperature=0.3,
            num_predict=600
        )
        
        if not text:
            logger.error(f"LLM returned no text for {symbol}")
            return self._fallback_response(symbol)
        
        # Try to extract JSON
        result = self._extract_json(text)
        
        # If JSON extraction fails, try natural language parsing
        if result is None:
            logger.info(f"JSON extraction failed, trying natural language parsing")
            result = self._parse_natural_language(text, symbol)
        
        # If still no result, try a simpler prompt
        if result is None:
            logger.info(f"Retrying with simpler prompt for {symbol}")
            result = self._retry_simple(symbol, technical_scores, fundamental_scores)
        
        if result is None:
            logger.error(f"All extraction methods failed for {symbol}")
            return self._fallback_response(symbol)
        
        # Validate output
        result = self._validate_output(result)
        logger.info(f"LLM synthesis complete for {symbol}")
        return result
    
    def _retry_simple(self, symbol: str, tech_scores: Dict, fund_scores: Dict) -> Optional[Dict]:
        """Retry with ultra-simple prompt."""
        tech_score = tech_scores.get('total', 50)
        fund_score = fund_scores.get('total', 50)
        
        simple_prompt = f"""Analyze {symbol}. Technical: {tech_score}/100. Fundamental: {fund_score}/100.

You MUST respond with ONLY this JSON (no other text):
{{"analysis_summary": "2 sentence analysis", "key_risks": ["risk1"], "key_catalysts": ["catalyst1"], "confidence_level": 0.6}}"""
        
        try:
            text = self._call_llm(
                prompt=simple_prompt,
                temperature=0.2,
                num_predict=400
            )
            
            if text:
                return self._extract_json(text)
        except Exception as e:
            logger.error(f"Simple retry failed: {e}")
        
        return None
    
    def _build_prompt(self, symbol, company_name, technical, tech_scores,
                    fundamental, fund_scores, news_context) -> str:
        
        # Get actual values with proper defaults
        current_price = getattr(technical, 'current_price', 0) if technical else 0
        rsi = getattr(technical, 'rsi', 50) if technical else 50
        trend = getattr(technical, 'trend', None)
        trend_str = trend.value if hasattr(trend, 'value') else str(trend) if trend else "neutral"
        
        pe = getattr(fundamental, 'pe_ratio', None) if fundamental else None
        roce = getattr(fundamental, 'roce', None) if fundamental else None
        promoter = getattr(fundamental, 'promoter_holding', None) if fundamental else None
        
        tech_score = tech_scores.get('total', 50)
        fund_score = fund_scores.get('total', 50)
        
        # Build news context properly
        if news_context and news_context.strip() and news_context != "None":
            news_section = f"Recent News Summary: {news_context[:150]}"
        else:
            news_section = "No recent news data available."
        
        prompt = f"""STOCK ANALYSIS: {symbol} ({company_name})

    DATA PROVIDED:
    - Technical Score: {tech_score}/100
    - Current Price: ₹{current_price}
    - RSI: {rsi} ({'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'})
    - Trend: {trend_str}
    - Fundamental Score: {fund_score}/100
    - P/E Ratio: {pe if pe else 'not available'}
    - ROCE: {roce if roce else 'not available'}%
    - Promoter Holding: {promoter if promoter else 'not available'}%
    - {news_section}

    TASK: Based on the data above, analyze {symbol} for investment.

    Provide your analysis as JSON:
    {{"analysis_summary": "2-3 sentences analyzing the stock based on the data", "key_risks": ["specific risk 1", "risk 2"], "key_catalysts": ["specific catalyst 1", "catalyst 2"], "confidence_level": 0.6}}"""
        
        return prompt
    
    def _validate_output(self, result: Dict) -> Dict:
        """Validate and sanitize output."""
        if 'analysis_summary' not in result or not result['analysis_summary']:
            result['analysis_summary'] = "Analysis summary unavailable."
        
        result['analysis_summary'] = str(result['analysis_summary'])[:500]
        
        if not isinstance(result.get('key_risks'), list):
            result['key_risks'] = ["Unable to determine specific risks"]
        result['key_risks'] = [str(r)[:150] for r in result['key_risks'][:5] if r]
        
        if not isinstance(result.get('key_catalysts'), list):
            result['key_catalysts'] = ["Unable to determine specific catalysts"]
        result['key_catalysts'] = [str(c)[:150] for c in result['key_catalysts'][:5] if c]
        
        if 'risk_flags' not in result or not isinstance(result['risk_flags'], dict):
            result['risk_flags'] = {
                "promoter_pledge_risk": False,
                "fii_outflow_risk": False,
                "liquidity_risk": False,
                "regulatory_risk": "",
                "corporate_governance_risk": ""
            }
        
        confidence = result.get('confidence_level', 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        result['confidence_level'] = max(0.0, min(1.0, float(confidence)))
        
        return result
    
    def _fallback_response(self, symbol: str) -> Dict:
        """Fallback response."""
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
            "confidence_level": 0.3
        }