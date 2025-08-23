"""
MarketSage - AI Stock Prediction Streamlit App
Loads pre-trained hybrid models for stock prediction and analysis
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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

# -------------------------
# Configuration & Setup
# -------------------------
st.set_page_config(
    layout="wide",
    page_title="MarketSage - AI Stock Predictor",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

try:
    load_dotenv(override=True)
    if not os.getenv('GEMINI_API_KEY'):
        st.sidebar.warning("⚠️ GEMINI_API_KEY not found in .env file")
except Exception as e:
    st.sidebar.error(f"❌ Error loading .env file: {str(e)}")

# Optional Gemini AI import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.sidebar.warning("⚠️ Google Generative AI not installed.")

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
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 0.5rem;
        text-align: center;
    }
    
    .gain-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .gain-negative {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    
    .model-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 MarketSage</h1>
    <h3>Advanced AI Stock Price Predictor</h3>
    <p>Powered by Pre-trained LSTM & Transformer Models</p>
</div>
""", unsafe_allow_html=True)

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
        # Load metadata
        with open(os.path.join(model_path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Load scaler
        with open(os.path.join(model_path, "scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
        
        return metadata, scaler
    except Exception as e:
        st.error(f"Error loading model data: {str(e)}")
        return None, None

def load_models(model_path, metadata):
    """Load LSTM and Transformer models"""
    try:
        # Load LSTM
        from keras.models import load_model
        lstm_model = load_model(os.path.join(model_path, "lstm_model.h5"), compile=False)
        lstm_model.compile(optimizer='adam', loss='mse')
        # Load Transformer
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
        # Get recent data (last 200 days to ensure we have enough after indicators)
        end_date = pd.Timestamp.today()
        start_date = end_date - timedelta(days=365)
        
        # Download stock data
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            return None, None
        
        df = df[['Open','High','Low','Close','Volume']]
        
        # Get sector data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector_symbol = get_sector_symbol(info.get("sector"))
        
        sector_df = yf.download(sector_symbol, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(sector_df.columns, pd.MultiIndex):
            sector_df.columns = sector_df.columns.get_level_values(0)
        sector_df = sector_df[['Close']].rename(columns={'Close':'Sector_Close'})
        
        # Merge
        df = df.join(sector_df, how="inner")
        
        # ADDED CODE TO CREATE 'is_business_day' AND HANDLE WEEKENDS/HOLIDAYS
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_range)
        
        # Forward-fill only OHLC
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].ffill()
        
        # Set Volume = 0 for non-business days
        df["Volume"] = df["Volume"].fillna(0)
        
        # Add business day flag (1 = trading day, 0 = weekend/holiday)
        df["is_business_day"] = df.index.to_series().map(lambda d: 1 if d.dayofweek < 5 else 0)
        df.index.name = "Date"
        
        # Add technical indicators (same as training)
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
        
        # Sector indicators
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
    
    # Prepare data
    scaled_data = scaler.transform(recent_data)
    
    # Get last sequence
    if len(scaled_data) < lookback_days:
        st.error(f"Not enough recent data. Need {lookback_days} days, got {len(scaled_data)}")
        return None
    
    last_sequence = scaled_data[-lookback_days:]
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_days):
        # Prepare input
        X_input = current_sequence.reshape(1, lookback_days, len(features))
        X_torch = torch.tensor(X_input, dtype=torch.float32)
        
        # Get predictions from both models
        lstm_pred_scaled = lstm_model.predict(X_input, verbose=0)[0, 0]
        transformer_pred = get_predictions(transformer_model, X_torch,
                                          scaler, len(features), device)[0]
        
        # Convert transformer prediction back to scaled
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = transformer_pred
        transformer_pred_scaled = scaler.transform(dummy)[0, 0]
        
        # Hybrid prediction (scaled)
        hybrid_pred_scaled = hybrid_weight * lstm_pred_scaled + (1-hybrid_weight) * transformer_pred_scaled
        
        # Store actual price prediction
        dummy[0, 0] = hybrid_pred_scaled
        hybrid_pred_price = scaler.inverse_transform(dummy)[0, 0]
        predictions.append(hybrid_pred_price)
        
        # Update sequence for next prediction (simplified approach)
        # In reality, we'd need to predict all features or use a different approach
        new_row = current_sequence[-1].copy()
        new_row[0] = hybrid_pred_scaled  # Update close price
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(predictions)

def calculate_recommendation_score(df, r2_score, future_preds, current_price):
    """Calculate investment recommendation score"""
    if df is None or future_preds is None:
        return 0, ["Insufficient data for recommendation"]
        
    score = 0
    factors = []
    
    # Technical factors
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
    
    # MACD analysis
    current_macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    if current_macd > macd_signal:
        score += 1
        factors.append("📈 MACD bullish crossover (+1)")
    else:
        score -= 1
        factors.append("📉 MACD bearish signal (-1)")
    
    # Moving average trend
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
    
    # Model confidence factor
    if r2_score > 0.8:
        score += 1
        factors.append(f"🎯 High model confidence (R²={r2_score:.3f}) (+1)")
    elif r2_score < 0.5:
        score -= 1
        factors.append(f"⚠️ Low model confidence (R²={r2_score:.3f}) (-1)")
    
    # Future prediction trend
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
# Sidebar Configuration
# -------------------------
st.sidebar.markdown("## 🎯 Model Selection")

# Get available models
available_models = get_available_models()

if not available_models:
    st.sidebar.error("❌ No trained models found!")
    st.sidebar.markdown("""
    **To get started:**
    1. Run `python train_hybrid_model.py` to train models
    2. Trained models will be saved in `saved_models/` directory
    3. Refresh this app to see available models
    """)
    st.stop()

# Model selection
model_options = [f"{model['symbol']} - {model['company_name']} (R²: {model['r2_score']:.3f})"
                 for model in available_models]

selected_idx = st.sidebar.selectbox("📊 Choose Trained Model",
                                    range(len(model_options)),
                                    format_func=lambda x: model_options[x])

selected_model = available_models[selected_idx]
st.sidebar.success(f"✅ Model loaded: {selected_model['symbol']}")

# Display model info
with st.sidebar.expander("📋 Model Details"):
    st.write(f"**Company:** {selected_model['company_name']}")
    st.write(f"**Sector:** {selected_model['sector']}")
    st.write(f"**Trained:** {selected_model['training_date'][:10]}")
    st.write(f"**R² Score:** {selected_model['r2_score']:.4f}")

# Prediction settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 Prediction Settings")
future_days = st.sidebar.slider("Days to Predict", 1, 90, 30)
show_confidence = st.sidebar.checkbox("Show Technical Analysis", True)
show_charts = st.sidebar.checkbox("Show Charts", True)

# Gemini AI Configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Insights")

if GEMINI_AVAILABLE:
    gemini_api_key = os.getenv('GEMINI_API_KEY', '')
    
    if not gemini_api_key:
        gemini_api_key = st.sidebar.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key for AI insights"
        )
    else:
        st.sidebar.success("✅ Gemini API key loaded from .env")
    
    enable_gemini = st.sidebar.checkbox("Enable AI Insights", value=bool(gemini_api_key))

    if gemini_api_key and enable_gemini:
        try:
            genai.configure(api_key=gemini_api_key)
            st.sidebar.success("🤖 AI Insights Ready!")
        except Exception as e:
            st.sidebar.error(f"❌ API error: {str(e)}")
            enable_gemini = False
else:
    enable_gemini = False

# -------------------------
# Main App Logic
# -------------------------
try:
    # Load model data
    metadata, scaler = load_model_data(selected_model['path'])
    if metadata is None:
        st.error("Failed to load model data")
        st.stop()

    # Load models
    with st.spinner("🤖 Loading AI models..."):
        lstm_model, transformer_model, device = load_models(selected_model['path'], metadata)
        if lstm_model is None:
            st.error("Failed to load models")
            st.stop()

    # Prepare recent data
    with st.spinner("📊 Preparing recent market data..."):
        recent_data, info = prepare_recent_data(
            selected_model['symbol'],
            metadata['lookback_days'],
            metadata['features']
        )
        
        if recent_data is None:
            st.error("Failed to fetch recent data")
            st.stop()

    # Make predictions
    with st.spinner("🔮 Generating predictions..."):
        predictions = predict_future(
            lstm_model, transformer_model, recent_data,
            scaler, metadata, device, future_days
        )
        
        if predictions is None:
            st.error("Failed to generate predictions")
            st.stop()

    # Current price and basic info
    current_price = recent_data['Close'].iloc[-1]
    
    # Display results
    st.markdown("## 📊 Current Market Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{current_price:.2f}",
            delta=f"{((current_price - recent_data['Close'].iloc[-2]) / recent_data['Close'].iloc[-2] * 100):.2f}%"
        )
    
    with col2:
        st.metric(
            label="RSI",
            value=f"{recent_data['RSI'].iloc[-1]:.1f}",
            delta="Oversold" if recent_data['RSI'].iloc[-1] < 30 else "Overbought" if recent_data['RSI'].iloc[-1] > 70 else "Normal"
        )
    
    with col3:
        st.metric(
            label="Model R² Score",
            value=f"{metadata['model_metrics']['hybrid']['r2']:.4f}",
            delta=f"RMSE: {metadata['model_metrics']['hybrid']['rmse']:.2f}"
        )
    
    with col4:
        volume_change = (recent_data['Volume'].iloc[-1] - recent_data['Volume'].iloc[-2]) / recent_data['Volume'].iloc[-2] * 100
        st.metric(
            label="Volume",
            value=f"{recent_data['Volume'].iloc[-1]:,.0f}",
            delta=f"{volume_change:.1f}%"
        )

    # Prediction results
    st.markdown("## 🔮 AI Predictions")
    
    periods = {
        "1 Week": 7,
        "2 Weeks": 14,
        "1 Month": 30,
        "2 Months": 60,
        "3 Months": 90
    }
    
    pred_cols = st.columns(len(periods))
    
    for idx, (period_name, days) in enumerate(periods.items()):
        if days <= future_days:
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

    # Calculate recommendation
    rec_score, rec_factors = calculate_recommendation_score(
        recent_data, metadata['model_metrics']['hybrid']['r2'],
        predictions, current_price
    )

    # Recommendation section
    st.markdown("## 🎯 AI Recommendation")
    
    rec_col1, rec_col2 = st.columns([1, 2])
    
    with rec_col1:
        if rec_score >= 3:
            recommendation = "🟢 STRONG BUY"
            rec_color = "#00ff00"
        elif rec_score >= 1:
            recommendation = "🔵 BUY"
            rec_color = "#0080ff"
        elif rec_score >= -1:
            recommendation = "🟡 HOLD"
            rec_color = "#ffff00"
        elif rec_score >= -3:
            recommendation = "🟠 SELL"
            rec_color = "#ff8000"
        else:
            recommendation = "🔴 STRONG SELL"
            rec_color = "#ff0000"
        
        st.markdown(f"""
        <div class="metric-container">
            <h2>{recommendation}</h2>
            <h1>Score: {rec_score}/5</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("### 📋 Analysis Factors:")
        for factor in rec_factors:
            st.markdown(f"• {factor}")

    # Charts
    if show_charts:
        st.markdown("## 📈 Technical Analysis Charts")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Price & Moving Averages", "RSI", "MACD"),
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Historical prices
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['Close'],
                       name="Historical Price", line=dict(color="blue")),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['EMA20'],
                       name="EMA20", line=dict(color="orange", dash="dash")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['EMA50'],
                       name="EMA50", line=dict(color="red", dash="dash")),
            row=1, col=1
        )
        
        # Predictions
        future_dates = pd.date_range(start=recent_data.index[-1] + timedelta(days=1),
                                     periods=future_days, freq='D')
        fig.add_trace(
            go.Scatter(x=future_dates, y=predictions,
                       name="AI Prediction", line=dict(color="green", width=3)),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['RSI'],
                       name="RSI", line=dict(color="purple")),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['MACD'],
                       name="MACD", line=dict(color="blue")),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['MACD_Signal'],
                       name="MACD Signal", line=dict(color="red")),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=True,
                          title_text=f"{selected_model['symbol']} - Technical Analysis & AI Predictions")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

    # Technical indicators table
    if show_confidence:
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

    # AI Insights section
    if enable_gemini and GEMINI_AVAILABLE:
        st.markdown("## 🤖 AI Market Insights")
        
        with st.spinner("🧠 Generating AI insights..."):
            try:
                # Prepare data for Gemini
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
                
                # Get AI insights
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

    # Model performance
    with st.expander("🎯 Model Performance Details"):
        st.markdown("### Hybrid Model Metrics")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("R² Score", f"{metadata['model_metrics']['hybrid']['r2']:.4f}")
            st.metric("LSTM Weight", f"{metadata['hybrid_weight']:.3f}")
        
        with perf_col2:
            st.metric("RMSE", f"₹{metadata['model_metrics']['hybrid']['rmse']:.2f}")
            st.metric("Transformer Weight", f"{1-metadata['hybrid_weight']:.3f}")
        
        with perf_col3:
            st.metric("MAE", f"₹{metadata['model_metrics']['hybrid']['mae']:.2f}")
            st.metric("Lookback Days", metadata['lookback_days'])
        
        st.markdown("### Individual Model Performance")
        
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
            'MAE': [
                metadata['model_metrics']['lstm']['mae'],
                metadata['model_metrics']['transformer']['mae'],
                metadata['model_metrics']['hybrid']['mae']
            ]
        })
        
        st.dataframe(models_perf, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.markdown("**Troubleshooting Tips:**")
    st.markdown("1. Ensure you have trained models in the `saved_models/` directory")
    st.markdown("2. Check that all required packages are installed")
    st.markdown("3. Verify internet connection for fetching market data")
    st.markdown("4. Try refreshing the page")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 MarketSage - AI-Powered Stock Prediction System</p>
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)