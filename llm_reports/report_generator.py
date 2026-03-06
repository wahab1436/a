import json
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMReportGenerator:
    def __init__(self, api_key: str = "", use_fallback: bool = True):
        self.api_key = api_key or os.getenv("AIzaSyCGxm5Cni1jvuuJNjCpSeo2HzMUdn180C4")
        self.use_fallback = use_fallback
        self.client = None
        self.model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.client = genai
                logger.info("Google Gemini client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}. Using fallback.")
                self.client = None

    def generate_report(self, asset: str, metrics: dict) -> str:
        if self.client and self.model:
            return self._generate_cloud_report(asset, metrics)
        else:
            return self._generate_local_report(asset, metrics)

    def _generate_cloud_report(self, asset: str, metrics: dict) -> str:
        prompt = f"""
        Act as a Senior Financial Risk Analyst. 
        Generate a JSON report for {asset} based on the following metrics:
        Regime State: {metrics['regime']}
        Forecasted Volatility (7-day): {metrics['volatility']:.4f}
        Sentiment Index (-1 to 1): {metrics['sentiment']:.2f}
        Composite Risk Score (0-100): {metrics['risk_score']:.2f}
        Current Drawdown: {metrics['drawdown']:.2%}
        
        Return ONLY valid JSON with the following keys:
        - executive_summary
        - regime_analysis
        - volatility_outlook
        - risk_mitigation
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Cloud generation failed: {e}")
            return self._generate_local_report(asset, metrics)

    def _generate_local_report(self, asset: str, metrics: dict) -> str:
        risk_level = "HIGH" if metrics['risk_score'] > 70 else "MEDIUM" if metrics['risk_score'] > 40 else "LOW"
        
        report = {
            "executive_summary": f"Risk level for {asset} is currently {risk_level}.",
            "regime_analysis": f"Market is in Regime State {metrics['regime']}.",
            "volatility_outlook": f"Expected volatility is {metrics['volatility']:.4f}.",
            "risk_mitigation": "Consider hedging positions if risk score exceeds 70."
        }
        return json.dumps(report, indent=2)
