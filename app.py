# MarketSage - AI Stock Prediction Streamlit App
# Loads pre-trained hybrid models for stock prediction and analysis

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import pickle
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Optional Gemini AI import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# -------------------------
# Custom CSS for a Polished UI
# -------------------------
st.markdown("""
<style>
    /* Main container and text styling */
    .stApp {
        background-color: #f0f2f6; /* A soft, professional gray background */
        color: #333;
    }

    /* Professional header with subtle gradients and shadow */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: bold;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .main-header h3 {
        font-size: 1.5rem;
        font-weight: 300;
        margin-top: 0;
    }
    .main-header p {
        font-size: 1rem;
        font-style: italic;
        opacity: 0.8;
    }
    
    /* Metric cards styling */
    .stMetric > div:nth-child(1) {
        border-radius: 10px;
        background-color: #ffffff;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #2a5298;
    }
    .stMetric label {
        color: #667eea;
        font-weight: bold;
    }
    .stMetric > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) {
        font-size: 1.5rem;
        color: #1e3c72;
    }
    .stMetric > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) {
        font-size: 1rem;
    }

    /* Prediction card styling with clear color coding */
    .prediction-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        color: #333;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 0.5rem;
        text-align: center;
        transition: transform 0.3s ease-in-out;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .prediction-card h4 {
        color: #2a5298;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    .prediction-card h2 {
        font-size: 2rem;
        margin: 0.25rem 0;
        color: #1e3c72;
    }
    .prediction-card h3 {
        font-size: 1.25rem;
        margin: 0.25rem 0;
    }

    /* Color-coded status for predictions */
    .gain-positive {
        border-left: 5px solid #28a745; /* Green border for gains */
    }
    .gain-negative {
        border-left: 5px solid #dc3545; /* Red border for losses */
    }

    /* Recommendation card styling */
    .recommendation-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
    }

    .recommendation-card h2 {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .recommendation-card h1 {
        font-size: 3.5rem;
        font-weight: bold;
        margin-top: 0;
    }
    
    .score-strong-buy { color: #28a745; }
    .score-buy { color: #17a2b8; }
    .score-hold { color: #ffc107; }
    .score-sell { color: #fd7e14; }
    .score-strong-sell { color: #dc3545; }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Model Definitions (same as training script)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int = 64, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(feature_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])

# -------------------------
# Utility Functions
# -------------------------
@st.cache_data
def get_available_models():
    """Get list of available trained models"""
    models_dir = "saved_models"
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    for symbol_dir in os.listdir(models_dir):
        metadata_path = os.path.join(models_dir, symbol_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                available_models.append({
                    'symbol': metadata['symbol'],
                    'company_name': metadata['company_name'],
                    'sector': metadata['sector'],
                    'training_date': metadata['training_date'],
                    'r2_score': metadata['model_metrics']['hybrid']['r2'],
                    'path': os.path.join(models_dir, symbol_dir)
                })
            except Exception as e:
                continue
    
    return available_models

def get_sector_symbol(sector):
    """Map sector to Indian market index"""
    sector_map = {
        "Financial Services": "^NSEBANK",
        "Banks": "^NSEBANK",
        "Technology": "^CNXIT",
        "Energy": "^CNXENERGY",
        "Consumer Defensive": "^CNXFMCG",
        "Healthcare": "^CNXPHARMA",
        "Pharmaceuticals": "^CNXPHARMA",
        "Consumer Cyclical": "^CNXAUTO",
        "Basic Materials": "^CNXMETAL",
        "Real Estate": "^CNXREALTY"
    }
    return sector_map.get(sector, "^NSEI")

@st.cache_data
def load_model_data(model_path):
    """Load trained model and metadata"""
    try:
        with open(os.path.join(model_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        with open(os.path.join(model_path, "scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
        return metadata, scaler
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None

def load_models(model_path, metadata):
    """Load LSTM and Transformer models"""
    try:
        from keras.models import load_model
        lstm_model = load_model(os.path.join(model_path, "lstm_model.h5"), compile=False)
        lstm_model.compile(optimizer='adam', loss='mse')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer_model = TransformerModel(
            feature_size=metadata['feature_size'],
            hidden_dim=metadata['model_architecture']['transformer']['hidden_dim'],
            num_heads=metadata['model_architecture']['transformer']['num_heads'],
            num_layers=metadata['model_architecture']['transformer']['num_layers'],
            dropout=metadata['model_architecture']['transformer']['dropout']
        ).to(device)
        
        transformer_model.load_state_dict(torch.load(
            os.path.join(model_path, "transformer_model.pth"),
            map_location=device
        ))
        
        return lstm_model, transformer_model, device
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

@st.cache_data
def prepare_recent_data(symbol, lookback_days, features):
    """Prepare recent data for prediction"""
    try:
        end_date = pd.Timestamp.today()
        start_date = end_date - timedelta(days=365)
        
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            return None, None
        
        df = df[['Open','High','Low','Close','Volume']]
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector_symbol = get_sector_symbol(info.get("sector"))
        
        sector_df = yf.download(sector_symbol, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(sector_df.columns, pd.MultiIndex):
            sector_df.columns = sector_df.columns.get_level_values(0)
        sector_df = sector_df[['Close']].rename(columns={'Close':'Sector_Close'})
        
        df = df.join(sector_df, how="inner")
        
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_range)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].ffill()
        
        df["Volume"] = df["Volume"].fillna(0)
        df["is_business_day"] = df.index.to_series().map(lambda d: 1 if d.dayofweek < 5 else 0)
        df.index.name = "Date"
        
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['Sector_EMA20'] = df['Sector_Close'].ewm(span=20, adjust=False).mean()
        df['Sector_EMA50'] = df['Sector_Close'].ewm(span=50, adjust=False).mean()
        sector_delta = df['Sector_Close'].diff()
        sector_gain = (sector_delta.where(sector_delta > 0, 0)).rolling(14).mean()
        sector_loss = (-sector_delta.where(sector_delta < 0, 0)).rolling(14).mean()
        sector_rs = sector_gain / sector_loss
        df['Sector_RSI'] = 100 - (100 / (1 + sector_rs))
        sector_ema12 = df['Sector_Close'].ewm(span=12, adjust=False).mean()
        sector_ema26 = df['Sector_Close'].ewm(span=26, adjust=False).mean()
        df['Sector_MACD'] = sector_ema12 - sector_ema26
        df['Sector_MACD_Signal'] = df['Sector_MACD'].ewm(span=9, adjust=False).mean()
        
        df.dropna(inplace=True)
        
        return df[features], info
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None

def inverse_transform(preds, scaler, n_features, target_col=0):
    """Inverse transform predictions"""
    dummy = np.zeros((len(preds), n_features))
    dummy[:, target_col] = preds.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col]

def get_predictions(model, X_torch, scaler, n_features, device, recalibrate=False):
    """Get predictions from Transformer model"""
    model.eval()
    with torch.no_grad():
        preds = model(X_torch.to(device)).cpu().numpy()
    preds_rescaled = inverse_transform(preds, scaler, n_features)
    return preds_rescaled

def predict_future(lstm_model, transformer_model, recent_data, scaler, metadata, device, future_days=30):
    """Predict future prices"""
    lookback_days = metadata['lookback_days']
    features = metadata['features']
    hybrid_weight = metadata['hybrid_weight']
    
    scaled_data = scaler.transform(recent_data)
    
    if len(scaled_data) < lookback_days:
        st.error(f"Not enough recent data. Need {lookback_days} days, got {len(scaled_data)}")
        return None
    
    last_sequence = scaled_data[-lookback_days:]
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        X_input = current_sequence.reshape(1, lookback_days, len(features))
        X_torch = torch.tensor(X_input, dtype=torch.float32)
        
        lstm_pred_scaled = lstm_model.predict(X_input, verbose=0)[0, 0]
        transformer_pred = get_predictions(transformer_model, X_torch,
                                         scaler, len(features), device)[0]
        
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = transformer_pred
        transformer_pred_scaled = scaler.transform(dummy)[0, 0]
        
        hybrid_pred_scaled = hybrid_weight * lstm_pred_scaled + (1-hybrid_weight) * transformer_pred_scaled
        
        dummy[0, 0] = hybrid_pred_scaled
        hybrid_pred_price = scaler.inverse_transform(dummy)[0, 0]
        predictions.append(hybrid_pred_price)
        
        new_row = current_sequence[-1].copy()
        new_row[0] = hybrid_pred_scaled
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(predictions)

def calculate_recommendation_score(df, r2_score, future_preds, current_price):
    """Calculate investment recommendation score"""
    if df is None or future_preds is None:
        return 0, ["Insufficient data for recommendation"]
        
    score = 0
    factors = []
    
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi < 30:
        score += 2
        factors.append("📊 RSI indicates oversold - potential buying opportunity (+2)")
    elif current_rsi > 70:
        score -= 2
        factors.append("📊 RSI indicates overbought - potential selling pressure (-2)")
    elif 40 <= current_rsi <= 60:
        score += 1
        factors.append("📊 RSI in healthy range (+1)")
    
    current_macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    if current_macd > macd_signal:
        score += 1
        factors.append("📈 MACD bullish crossover (+1)")
    else:
        score -= 1
        factors.append("📉 MACD bearish signal (-1)")
    
    current_price = df['Close'].iloc[-1]
    ema20 = df['EMA20'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    
    if current_price > ema20 > ema50:
        score += 2
        factors.append("📈 Strong uptrend - price above both EMAs (+2)")
    elif current_price > ema20:
        score += 1
        factors.append("📈 Mild uptrend - price above EMA20 (+1)")
    elif current_price < ema20 < ema50:
        score -= 2
        factors.append("📉 Strong downtrend - price below both EMAs (-2)")
    else:
        score -= 1
        factors.append("📉 Mild downtrend - price below EMA20 (-1)")
    
    if r2_score > 0.8:
        score += 1
        factors.append(f"🎯 High model confidence (R²={r2_score:.3f}) (+1)")
    elif r2_score < 0.5:
        score -= 1
        factors.append(f"⚠️ Low model confidence (R²={r2_score:.3f}) (-1)")
    
    if len(future_preds) >= 7:
        week_gain = (future_preds[6] - current_price) / current_price * 100
        if week_gain > 5:
            score += 1
            factors.append(f"🚀 Strong positive outlook (+1)")
        elif week_gain < -5:
            score -= 1
            factors.append(f"⚠️ Negative outlook (-1)")
    
    return score, factors

def calculate_gain_loss(predictions, current_price, days):
    """Calculate gain/loss percentage for given days"""
    if days > len(predictions):
        return 0, "Insufficient predictions", current_price
        
    future_price = predictions[days-1]
    gain_pct = ((future_price - current_price) / current_price) * 100
    
    if gain_pct > 0:
        status = "GAIN"
    else:
        status = "LOSS"
    
    return gain_pct, status, future_price

# -------------------------
# UI Rendering Functions
# -------------------------

def render_header(symbol, company_name):
    """Renders the main application header."""
    st.markdown(f"""
    <div class="main-header">
        <h1>📈 MarketSage</h1>
        <h3>AI-Powered Stock Predictor for {company_name} ({symbol})</h3>
        <p>Built with Pre-trained LSTM & Transformer Models</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(available_models):
    """Configures the sidebar with model selection and settings."""
    st.sidebar.markdown("## 🎯 Model & Settings")
    model_options = [f"{model['symbol']} - {model['company_name']} (R²: {model['r2_score']:.3f})"
                     for model in available_models]

    selected_idx = st.sidebar.selectbox("📊 Choose Trained Model",
                                        range(len(model_options)),
                                        format_func=lambda x: model_options[x],
                                        help="Select a model trained on a specific stock.")

    selected_model = available_models[selected_idx]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Prediction & Display")
    future_days = st.sidebar.slider("Days to Predict", 1, 90, 30, help="Choose how many days into the future to predict.")
    show_technical_details = st.sidebar.checkbox("Show Technical Details", True, help="Display a detailed table of technical indicators.")
    show_charts = st.sidebar.checkbox("Show Charts", True, help="Display interactive charts for analysis.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Insights")
    enable_gemini = False
    if GEMINI_AVAILABLE:
        gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        if not gemini_api_key:
            gemini_api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password",
                                                   help="Enter your Google Gemini API key to enable AI insights.")
            if gemini_api_key:
                os.environ['GEMINI_API_KEY'] = gemini_api_key
        
        if gemini_api_key:
            st.sidebar.success("✅ Gemini API key loaded")
            enable_gemini = st.sidebar.checkbox("Enable AI Insights", value=True)
            if enable_gemini:
                try:
                    genai.configure(api_key=gemini_api_key)
                except Exception as e:
                    st.sidebar.error(f"❌ API key error: {str(e)}")
                    enable_gemini = False
        else:
            st.sidebar.warning("⚠️ No Gemini API key provided. AI insights are disabled.")

    return selected_model, future_days, show_technical_details, show_charts, enable_gemini

def render_metrics(current_price, recent_data, metadata):
    """Renders the top-level metric cards."""
    st.markdown("## 📊 Current Market Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Current Price", value=f"₹{current_price:.2f}",
                  delta=f"{((current_price - recent_data['Close'].iloc[-2]) / recent_data['Close'].iloc[-2] * 100):.2f}%")
    
    with col2:
        st.metric(label="RSI", value=f"{recent_data['RSI'].iloc[-1]:.1f}",
                  delta="Oversold" if recent_data['RSI'].iloc[-1] < 30 else "Overbought" if recent_data['RSI'].iloc[-1] > 70 else "Normal")
    
    with col3:
        st.metric(label="Model R² Score", value=f"{metadata['model_metrics']['hybrid']['r2']:.4f}",
                  delta=f"RMSE: {metadata['model_metrics']['hybrid']['rmse']:.2f}")
    
    with col4:
        volume_change = (recent_data['Volume'].iloc[-1] - recent_data['Volume'].iloc[-2]) / recent_data['Volume'].iloc[-2] * 100 if recent_data['Volume'].iloc[-2] != 0 else 0
        st.metric(label="Volume", value=f"{recent_data['Volume'].iloc[-1]:,.0f}",
                  delta=f"{volume_change:.1f}%")

def render_predictions(predictions, current_price, future_days):
    """Renders the AI prediction cards."""
    st.markdown("## 🔮 AI Predictions")
    # Define a new, more detailed set of periods
    periods = {
        "1 Day": 1, 
        "1 Week": 7, 
        "2 Weeks": 14, 
        "1 Month": 30, 
        "3 Months": 90, 
        "6 Months": 180  # Added 6-month prediction
    }
    
    # Filter periods based on the user's selected future_days
    periods_to_display = {
        name: days for name, days in periods.items() if days <= future_days
    }
    
    pred_cols = st.columns(len(periods_to_display))
    
    for idx, (period_name, days) in enumerate(periods_to_display.items()):
        gain_pct, status, pred_price = calculate_gain_loss(predictions, current_price, days)
        
        with pred_cols[idx]:
            color_class = "gain-positive" if gain_pct > 0 else "gain-negative"
            st.markdown(f"""
            <div class="prediction-card {color_class}">
                <h4>{period_name}</h4>
                <h2>₹{pred_price:.2f}</h2>
                <h3>{gain_pct:+.2f}%</h3>
                <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)

def render_recommendation(rec_score, rec_factors):
    """Renders the AI recommendation and analysis factors."""
    st.markdown("## 🎯 AI Recommendation")
    rec_col1, rec_col2 = st.columns([1, 2])
    
    if rec_score >= 3:
        recommendation = "STRONG BUY"
        rec_class = "score-strong-buy"
    elif rec_score >= 1:
        recommendation = "BUY"
        rec_class = "score-buy"
    elif rec_score >= -1:
        recommendation = "HOLD"
        rec_class = "score-hold"
    elif rec_score >= -3:
        recommendation = "SELL"
        rec_class = "score-sell"
    else:
        recommendation = "STRONG SELL"
        rec_class = "score-strong-sell"

    with rec_col1:
        st.markdown(f"""
        <div class="recommendation-card">
            <h2 class="{rec_class}">{recommendation}</h2>
            <h1>Score: {rec_score}/5</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("### Analysis Factors:")
        for factor in rec_factors:
            st.markdown(f"• {factor}")

def render_charts(recent_data, predictions, future_days, symbol):
    """Renders interactive Plotly charts."""
    st.markdown("## 📈 Technical Analysis Charts")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Price & Moving Averages", "RSI", "MACD"),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Historical prices
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], name="Historical Price", line=dict(color="#2a5298")), row=1, col=1)
    
    # Add a dotted line connecting historical and predicted data
    connector_dates = [recent_data.index[-1], pd.date_range(start=recent_data.index[-1] + timedelta(days=1), periods=1)[0]]
    connector_prices = [recent_data['Close'].iloc[-1], predictions[0]]
    fig.add_trace(go.Scatter(x=connector_dates, y=connector_prices, 
                             mode='lines', name="Prediction Start", 
                             line=dict(color="#28a745", width=2, dash="dash")), row=1, col=1)
    
    # Predictions
    future_dates = pd.date_range(start=recent_data.index[-1] + timedelta(days=1), periods=future_days, freq='D')
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, name="AI Prediction", line=dict(color="#28a745", width=3)), row=1, col=1)

    # Moving averages, RSI, and MACD traces remain the same
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['EMA20'], name="EMA20", line=dict(color="#ffc107", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['EMA50'], name="EMA50", line=dict(color="#dc3545", dash="dash")), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['RSI'], name="RSI", line=dict(color="#667eea")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc3545", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="#28a745", row=2, col=1, annotation_text="Oversold")
    
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD'], name="MACD", line=dict(color="#17a2b8")), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD_Signal'], name="MACD Signal", line=dict(color="#ffc107")), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True,
                      title_text=f"<b>{symbol} Technical Analysis & AI Predictions</b>")
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_layout(template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)

def render_technical_table(recent_data, current_price):
    """Renders a detailed table of technical indicators."""
    st.markdown("## 📊 Current Technical Indicators")
    tech_data = {
        "Indicator": ["RSI", "MACD", "MACD Signal", "EMA20", "EMA50", "Current Price"],
        "Value": [
            f"{recent_data['RSI'].iloc[-1]:.2f}",
            f"{recent_data['MACD'].iloc[-1]:.4f}",
            f"{recent_data['MACD_Signal'].iloc[-1]:.4f}",
            f"₹{recent_data['EMA20'].iloc[-1]:.2f}",
            f"₹{recent_data['EMA50'].iloc[-1]:.2f}",
            f"₹{current_price:.2f}"
        ],
        "Signal": [
            "Oversold" if recent_data['RSI'].iloc[-1] < 30 else "Overbought" if recent_data['RSI'].iloc[-1] > 70 else "Neutral",
            "Bullish" if recent_data['MACD'].iloc[-1] > recent_data['MACD_Signal'].iloc[-1] else "Bearish",
            "-",
            "Support" if current_price > recent_data['EMA20'].iloc[-1] else "Resistance",
            "Support" if current_price > recent_data['EMA50'].iloc[-1] else "Resistance",
            "-"
        ]
    }
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True)

def render_gemini_insights(selected_model, current_price, predictions, recent_data, metadata, rec_score, future_days):
    """Generates and renders insights using the Gemini API."""
    st.markdown("## 🤖 AI Market Insights")
    with st.spinner("🧠 Generating AI insights..."):
        try:
            stock_data = {
                'symbol': selected_model['symbol'],
                'company_name': selected_model['company_name'],
                'sector': selected_model['sector'],
                'current_price': current_price,
                'predictions': {
                    '1 Week': {
                        'gain_pct': calculate_gain_loss(predictions, current_price, 7)[0] if future_days >= 7 else 0,
                        'predicted_price': calculate_gain_loss(predictions, current_price, 7)[2] if future_days >= 7 else current_price
                    },
                    '1 Month': {
                        'gain_pct': calculate_gain_loss(predictions, current_price, 30)[0] if future_days >= 30 else 0,
                        'predicted_price': calculate_gain_loss(predictions, current_price, 30)[2] if future_days >= 30 else current_price
                    }
                },
                'technical_indicators': {
                    'RSI': recent_data['RSI'].iloc[-1],
                    'MACD': recent_data['MACD'].iloc[-1],
                    'MACD_Signal': recent_data['MACD_Signal'].iloc[-1],
                    'EMA20': recent_data['EMA20'].iloc[-1],
                    'EMA50': recent_data['EMA50'].iloc[-1]
                },
                'model_accuracy': metadata['model_metrics']['hybrid']['r2'],
                'recommendation_score': rec_score
            }
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            As a senior financial analyst, provide concise investment insights for {stock_data['company_name']} ({stock_data['symbol']}).

            **Current Status:**
            - Price: ₹{stock_data['current_price']:.2f}
            - Sector: {stock_data['sector']}
            - AI Model Accuracy: {stock_data['model_accuracy']:.3f}

            **AI Predictions:**
            - 1 Week: {stock_data['predictions']['1 Week']['gain_pct']:.1f}%
            - 1 Month: {stock_data['predictions']['1 Month']['gain_pct']:.1f}%

            **Technical Indicators:**
            - RSI: {stock_data['technical_indicators']['RSI']:.1f}
            - MACD vs Signal: {stock_data['technical_indicators']['MACD']:.4f} vs {stock_data['technical_indicators']['MACD_Signal']:.4f}

            **Recommendation Score:** {stock_data['recommendation_score']}/5

            Provide a brief analysis covering:
            1. **Key Insights** (2-3 bullet points)
            2. **Risk Factors** (main concerns)
            3. **Investment Outlook** (short-term view)

            Keep it concise and actionable for retail investors.
            """
            
            response = model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error generating AI insights: {str(e)}")

# -------------------------
# Main Application Flow
# -------------------------

def main():
    st.set_page_config(layout="wide", page_title="MarketSage - AI Stock Predictor", page_icon="📈")
    
    load_dotenv(override=True)
    available_models = get_available_models()

    if not available_models:
        st.error("❌ No trained models found!")
        st.info("To get started, please train a model by running `python train_hybrid_model.py` and then refresh this page.")
        st.stop()

    selected_model, future_days, show_technical_details, show_charts, enable_gemini = render_sidebar(available_models)
    
    render_header(selected_model['symbol'], selected_model['company_name'])
    
    try:
        with st.spinner("🤖 Loading AI models and data..."):
            metadata, scaler = load_model_data(selected_model['path'])
            if metadata is None: st.stop()

            lstm_model, transformer_model, device = load_models(selected_model['path'], metadata)
            if lstm_model is None: st.stop()

            recent_data, info = prepare_recent_data(selected_model['symbol'], metadata['lookback_days'], metadata['features'])
            if recent_data is None: st.stop()

            predictions = predict_future(lstm_model, transformer_model, recent_data, scaler, metadata, device, future_days)
            if predictions is None: st.stop()

        current_price = recent_data['Close'].iloc[-1]
        rec_score, rec_factors = calculate_recommendation_score(recent_data, metadata['model_metrics']['hybrid']['r2'], predictions, current_price)

        render_metrics(current_price, recent_data, metadata)
        st.write("---")
        render_predictions(predictions, current_price, future_days)
        st.write("---")
        render_recommendation(rec_score, rec_factors)
        
        st.write("---")
        if show_charts:
            render_charts(recent_data, predictions, future_days, selected_model['symbol'])
        
        if show_technical_details:
            st.write("---")
            render_technical_table(recent_data, current_price)

        if enable_gemini and GEMINI_AVAILABLE:
            st.write("---")
            render_gemini_insights(selected_model, current_price, predictions, recent_data, metadata, rec_score, future_days)

        with st.expander("🎯 View Model Performance Details"):
            st.markdown("### Hybrid Model Metrics")
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("R² Score", f"{metadata['model_metrics']['hybrid']['r2']:.4f}")
            with perf_col2:
                st.metric("RMSE", f"₹{metadata['model_metrics']['hybrid']['rmse']:.2f}")
            with perf_col3:
                st.metric("MAE", f"₹{metadata['model_metrics']['hybrid']['mae']:.2f}")
            
            st.markdown("### Individual Model Breakdown")
            models_perf = pd.DataFrame({
                'Model': ['LSTM', 'Transformer', 'Hybrid'],
                'R² Score': [
                    metadata['model_metrics']['lstm']['r2'],
                    metadata['model_metrics']['transformer']['r2'],
                    metadata['model_metrics']['hybrid']['r2']
                ],
                'RMSE': [
                    metadata['model_metrics']['lstm']['rmse'],
                    metadata['model_metrics']['transformer']['rmse'],
                    metadata['model_metrics']['hybrid']['rmse']
                ],
            })
            st.dataframe(models_perf, use_container_width=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please ensure all models are correctly loaded and data fetching is successful.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📈 MarketSage - AI-Powered Stock Prediction System</p>
        <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()