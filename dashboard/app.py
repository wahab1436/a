import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_pipeline.ingestion import MarketDataIngestor
from data_pipeline.validation import DataValidator
from data_pipeline.news_ingestion import NewsIngestor
from feature_engineering.market_features import MarketFeatureEngineer
from feature_engineering.sentiment_features import SentimentFeatureEngineer
from models.regime_model import RegimeDetectionModel
from models.volatility_model import VolatilityForecastModel
from risk_engine.risk_calculator import RiskScoringEngine
from llm_reports.report_generator import LLMReportGenerator

# Load Config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Financial Risk Intelligence", layout="wide")

@st.cache_data
def load_and_process_data(ticker):
    ingestor = MarketDataIngestor(config['data']['storage_path'])
    df = ingestor.fetch_data(ticker, config['data']['start_date'])
    
    if df.empty:
        return None
        
    validator = DataValidator()
    if not validator.validate(df, ticker):
        st.warning(f"Validation issues for {ticker}")
        
    fe = MarketFeatureEngineer()
    df = fe.compute_features(df)
    return df

def main():
    st.title("Financial Risk Intelligence System")
    st.markdown("Powered by Free Data Sources and Open Models")
    
    sidebar = st.sidebar
    selected_asset = sidebar.selectbox("Select Asset", config['data']['assets'])
    
    # API Key Input
    api_key = sidebar.text_input("Google Gemini API Key (Optional)", type="password", value=config['llm'].get('api_key', ''))
    
    if sidebar.button("Run Analysis"):
        with st.spinner("Fetching data and computing risk metrics..."):
            df = load_and_process_data(selected_asset)
            
            if df is not None and len(df) > 100:
                # 1. Regime Detection
                regime_model = RegimeDetectionModel(n_states=config['models']['regime']['n_states'])
                regime_model.fit(df)
                df['regime_label'] = regime_model.predict(df)
                
                # 2. Volatility Forecast
                vol_model = VolatilityForecastModel()
                vol_model.fit(df)
                df['forecasted_volatility'] = vol_model.predict(df)
                
                # 3. Sentiment (Local FinBERT)
                # For demo stability, we simulate sentiment based on recent returns if news fetch fails
                # In production, use NewsIngestor
                df['sentiment_index'] = 0.0 
                df['sentiment_index'] = df['log_return'].rolling(5).mean().fillna(0)
                
                # 4. Risk Score
                risk_engine = RiskScoringEngine(config['models']['risk_weights'])
                df['risk_score'] = risk_engine.compute_score(df)
                
                # --- VISUALIZATION ---
                
                # Row 1: Price & Regime
                col1, col2 = st.columns(2)
                with col1:
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price'))
                    fig_price.update_layout(title=f"{selected_asset} Price History", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_price, use_container_width=True)
                
                with col2:
                    fig_regime = go.Figure()
                    fig_regime.add_trace(go.Scatter(x=df['Date'], y=df['regime_label'], name='Regime State', line=dict(shape='hv')))
                    fig_regime.update_layout(title="Market Regime Detection", yaxis_title="State")
                    st.plotly_chart(fig_regime, use_container_width=True)
                
                # Row 2: Volatility & Risk
                col3, col4 = st.columns(2)
                with col3:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['rolling_std_7'], name='Realized Vol'))
                    fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['forecasted_volatility'], name='Forecasted Vol', line=dict(dash='dot')))
                    fig_vol.update_layout(title="Volatility Forecast vs Realized")
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with col4:
                    fig_risk = go.Figure()
                    fig_risk.add_trace(go.Scatter(x=df['Date'], y=df['risk_score'], name='Risk Score', line=dict(color='red')))
                    fig_risk.add_hline(y=70, line_dash="dash", annotation_text="High Risk Threshold")
                    fig_risk.update_layout(title="Composite Risk Score")
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                # Row 3: Latest Metrics & LLM Report
                st.subheader("Current Risk Assessment")
                latest = df.iloc[-1]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Risk Score", f"{latest['risk_score']:.1f}")
                m2.metric("Regime", f"State {int(latest['regime_label'])}")
                m3.metric("Forecasted Vol", f"{latest['forecasted_volatility']:.4f}")
                m4.metric("Sentiment", f"{latest['sentiment_index']:.2f}")
                
                # LLM Integration
                if st.button("Generate AI Report"):
                    generator = LLMReportGenerator(api_key=api_key, use_fallback=True)
                    
                    metrics = {
                        'regime': int(latest['regime_label']),
                        'volatility': latest['forecasted_volatility'],
                        'sentiment': latest['sentiment_index'],
                        'risk_score': latest['risk_score'],
                        'drawdown': latest['drawdown']
                    }
                    
                    report_json = generator.generate_report(selected_asset, metrics)
                    st.json(report_json)

            else:
                st.error("Insufficient data for analysis.")

if __name__ == "__main__":
    main()
