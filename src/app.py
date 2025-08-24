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
import traceback

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
    /* ------------------------- */
    /* Global & Main Component Styles */
    /* ------------------------- */

    /* General App Container Styling (Light Mode) */
    .stApp {
        background-color: #f0f2f6;
        color: #333;
    }
    
    /* Professional Header with subtle gradients and shadow (consistent in both modes) */
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
    
    /* Current Market Status - Polished Metric Card Styling (Light Mode) */
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 1rem;
        font-weight: bold;
        color: #667eea;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #1e3c72;
    }
    .metric-delta {
        font-size: 1rem;
        font-weight: bold;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .positive-border {
        border-color: #28a745;
    }
    .negative-border {
        border-color: #dc3545;
    }
    .neutral-border {
        border-color: #ffc107;
    }

    /* Prediction Card Styling with clear color coding (Light Mode) */
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
    }
    .prediction-card h3 {
        font-size: 1.25rem;
        margin: 0.25rem 0;
    }
    .prediction-card p {
        font-weight: bold;
    }

    /* Color-coded borders and text for predictions */
    .gain-positive {
        border-left: 5px solid #28a745;
    }
    .gain-negative {
        border-left: 5px solid #dc3545;
    }
    .gain-text {
        color: #28a745;
    }
    .loss-text {
        color: #dc3545;
    }
    
    /* New Recommendation Card Styling (Light Mode) */
    .recommendation-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
    }

    .recommendation-title h1 {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .recommendation-score {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Recommendation Colors */
    .buy-color { color: #28a745; }
    .strong-buy-color { color: #11783f; }
    .hold-color { color: #ffc107; }
    .sell-color { color: #fd7e14; }
    .strong-sell-color { color: #dc3545; }

    /* Analysis Factors Styling (Light Mode) */
    .analysis-factors {
        list-style-type: none;
        padding: 0;
    }
    .analysis-factors li {
        background-color: #f8f9fa;
        margin-bottom: 10px;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .analysis-factors li span {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .score-pill {
        padding: 2px 8px;
        border-radius: 50px;
        font-weight: bold;
        color: white;
        font-size: 0.8rem;
        min-width: 40px;
        text-align: center;
    }
    .score-positive { background-color: #28a745; }
    .score-negative { background-color: #dc3545; }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: none;
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        font-weight: bold;
    }

    /* New styles for the 52-Week High/Low section */
    .slider-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 1rem;
    }
    .slider-label {
        font-size: 0.875rem;
        color: #6c757d;
    }
    .slider-value {
        font-size: 1.125rem;
        font-weight: bold;
        color: #333;
    }

    /* Custom horizontal divider */
    .st-emotion-cache-1px2j72 hr {
        border-top: 1px solid #e0e0e0;
    }

    /* ------------------------- */
    /* Dark Mode Overrides */
    /* ------------------------- */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #121212;
            color: #f0f2f6;
        }

        .metric-card {
            background-color: #1e1e1e;
            box-shadow: 0 4px 12px rgba(255, 255, 255, 0.05);
        }
        .metric-value {
            color: #f0f2f6;
        }
        .metric-title {
            color: #ccc;
        }

        .stMetric > div:nth-child(1) {
            background-color: #1e1e1e;
            border-left: 5px solid #667eea;
        }
        .stMetric label {
            color: #ccc;
        }
        .stMetric > div:nth-child(1) > div:nth-child(2) > div:nth-child(1) {
            color: #fff;
        }
        
        .prediction-card {
            background-color: #1e1e1e;
            color: #f0f2f6;
            box-shadow: 0 8px 25px rgba(255, 255, 255, 0.05);
        }
        .prediction-card h4 {
            color: #667eea;
        }
        .prediction-card h2 {
            color: #f0f2f6;
        }

        .recommendation-card {
            background-color: #1e1e1e;
            box-shadow: 0 8px 25px rgba(255, 255, 255, 0.05);
        }

        .analysis-factors li {
            background-color: #1e1e1e;
            border: 1px solid #333;
        }
        
        .streamlit-expanderHeader {
            background-color: #1e1e1e;
            box-shadow: 0 4px 8px rgba(255, 255, 255, 0.03);
            color: #f0f2f6;
        }
        
        .slider-value {
            color: #f0f2f6;
        }
        .slider-label {
            color: #ccc;
        }
        .st-emotion-cache-1px2j72 hr {
            border-top: 1px solid #444;
        }
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
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models"))
    if not os.path.exists(models_dir):
        return []
    
    available_models = []
    for symbol_dir in os.listdir(models_dir):
        metadata_path = os.path.join(models_dir, symbol_dir, "metadata.json")
        print(f"DEBUG: models_dir={models_dir}, symbol_dir={symbol_dir}, metadata_path={metadata_path}")
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
                    'path': os.path.join(models_dir, symbol_dir),
                    'model_features': metadata['features']
                })
            except Exception as e:
                print(f"DEBUG: Error loading {metadata_path}: {e}")
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
def prepare_recent_data(symbol):
    """
    Prepare recent data for prediction and display.
    
    This function returns the full DataFrame and the yfinance info dictionary.
    The specific features for the model will be selected later using metadata.
    """
    try:
        end_date = pd.Timestamp.today()
        start_date = end_date - timedelta(days=365)
        
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            return None, None
        
        df = df[['Open','High','Low','Close','Volume']].copy()
        
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
        
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()
        
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
        
        return df, info
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        st.error(traceback.format_exc())
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

def predict_future(lstm_model, transformer_model, features_for_model, scaler, metadata, device, future_days=30):
    """Predict future prices"""
    lookback_days = metadata['lookback_days']
    features = metadata['features']
    hybrid_weight = metadata['hybrid_weight']
    
    scaled_data = scaler.transform(features_for_model)
    
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
    """Calculate investment recommendation score with new factors"""
    if df is None or future_preds is None:
        return 0, ["Insufficient data for recommendation"]
    
    score = 0
    factors = []
    
    # 1. RSI Score
    current_rsi = df['RSI'].iloc[-1]
    if current_rsi < 30:
        score += 1.5
        factors.append("📊 RSI indicates oversold - potential buying opportunity (+1.5)")
    elif current_rsi > 70:
        score -= 1.5
        factors.append("📊 RSI indicates overbought - potential selling pressure (-1.5)")
    elif 40 <= current_rsi <= 60:
        score += 0.5
        factors.append("📊 RSI in healthy range (+0.5)")
    
    # 2. MACD Score
    current_macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    if current_macd > macd_signal:
        score += 1
        factors.append("📈 MACD bullish crossover (+1)")
    else:
        score -= 1
        factors.append("📉 MACD bearish signal (-1)")

    # 3. Volume Confirmation
    recent_volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].rolling(window=30).mean().iloc[-1]
    
    recent_price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
    
    if recent_price_change > 0 and recent_volume > avg_volume * 1.5:
        score += 1.5
        factors.append("💪 Strong volume confirms recent price increase (+1.5)")
    elif recent_price_change < 0 and recent_volume > avg_volume * 1.5:
        score -= 1.5
        factors.append("⚠️ High volume confirms recent price decrease (-1.5)")
    elif recent_price_change > 0 and recent_volume < avg_volume * 0.8:
        score -= 0.5
        factors.append("📉 Weak volume suggests recent price increase is fragile (-0.5)")
    
    # 4. Volatility (ATR) - Risk Factor
    current_atr = df['ATR'].iloc[-1]
    avg_atr = df['ATR'].mean()
    
    if current_atr < avg_atr * 0.8:
        score += 0.5
        factors.append("⚖️ Lower than average volatility (lower risk) (+0.5)")
    elif current_atr > avg_atr * 1.2:
        score -= 0.5
        factors.append("⚡ Higher than average volatility (higher risk) (-0.5)")

    # 5. Trend Strength (EMA)
    ema50 = df['EMA50'].iloc[-1]
    if current_price > ema50:
        score += 1
        factors.append("🚀 Price is above EMA50, indicating a clear long-term uptrend (+1)")
    elif current_price < ema50:
        score -= 1
        factors.append("🔻 Price is below EMA50, indicating a long-term downtrend (-1)")
    
    # 6. Model Confidence
    if r2_score > 0.8:
        score += 0.5
        factors.append(f"🎯 High model confidence (R²={r2_score:.3f}) (+0.5)")
    elif r2_score < 0.5:
        score -= 0.5
        factors.append(f"⚠️ Low model confidence (R²={r2_score:.3f}) (-0.5)")
    
    # 7. Short-term Prediction Outlook
    if len(future_preds) >= 7:
        week_gain = (future_preds[6] - current_price) / current_price * 100
        if week_gain > 5:
            score += 1
            factors.append(f"🚀 Strong positive short-term outlook from AI (+1)")
        elif week_gain < -5:
            score -= 1
            factors.append(f"⚠️ Negative short-term outlook from AI (-1)")
    
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
def render_welcome_page(available_models):
    """Renders the introductory welcome page."""
    st.markdown("""
        <div class="main-header" style="text-align: left;">
            <h1>👋 Welcome to MarketSage!</h1>
            <h3>Your AI-Powered Stock Prediction Assistant</h3>
            <p>Made by <a href="https://github.com/lazy-coder-03">Sayantan Ghosh (lazy-Coder)</a></p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        ### What is MarketSage?
        MarketSage is a tool that uses **advanced AI models** to analyze historical stock data and predict future price movements. It's designed to help you, even if you have no prior expertise in AI or financial analysis, get a better understanding of potential stock performance.

        ### How It Works
        1.  **Select a Stock:** Choose from a list of stocks for which a trained AI model is available.
        2.  **Define Prediction Days:** Pick how many days into the future you want the AI to predict.
        3.  **Get Your Report:** Click the button to generate a comprehensive report that includes:
            * **AI Predictions:** A forecast of the stock's price for the coming days.
            * **Technical Indicators:** Key metrics like RSI and MACD explained in simple terms.
            * **AI Recommendation:** A clear "Buy," "Hold," or "Sell" recommendation based on the combined analysis.
            * **AI Insights:** A detailed, easy-to-read summary from a large language model (like Gemini) to provide market context.

        ⚠️ **Disclaimer:** This tool is for **educational and informational purposes only**. The predictions are based on historical data and do not guarantee future performance. Please consult a qualified financial advisor before making any investment decisions.
    """)
    st.markdown("---")
    st.markdown("### 🚀 Get Started")

    # Call the API key check function in the welcome sidebar
    check_and_configure_gemini()

    model_options = [f"{model['symbol']} - {model['company_name']}" for model in available_models]
    selected_option = st.selectbox("📊 Choose a stock to analyze", model_options)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        future_days = st.slider("🔮 Days to Predict", 1, 90, 30, key="future_days_welcome", help="Choose how many days into the future the AI will predict.")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📈 Generate AI Report", use_container_width=True):
            selected_idx = model_options.index(selected_option)
            st.session_state.selected_symbol = available_models[selected_idx]['symbol']
            st.session_state.future_days = future_days
            st.session_state.page = 'main_app'
            st.rerun()


def render_header(symbol, company_name):
    """Renders the main application header."""
    st.markdown(f"""
    <div class="main-header">
        <h1>📈 MarketSage</h1>
        <h3>AI-Powered Stock Predictor for {company_name} ({symbol})</h3>
        <p>Built with Pre-trained LSTM & Transformer Models Made by <a href="https://github.com/lazy-coder-03">Sayantan Ghosh (lazy-Coder)</a></p>
    </div>
    """, unsafe_allow_html=True)

def check_and_configure_gemini():
    """Handles the Gemini API key input and validation in the sidebar."""
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("⚠️ Gemini library not found. AI insights disabled.")
        return False
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Insights")
    st.sidebar.info(
        "To enable AI Market Insights powered by Gemini, you need a valid API key. "
        "This feature is optional but highly recommended."
    )

    # Get the current key from the environment (may be from a .env file or a previous session input)
    current_api_key = os.getenv('GEMINI_API_KEY', '')
    
    is_key_valid = False
    
    # Check if a key exists and is valid
    if current_api_key:
        try:
            genai.configure(api_key=current_api_key)
            genai.get_model('gemini-1.5-flash')
            is_key_valid = True
        except Exception as e:
            is_key_valid = False

    # Display status and instructions based on the key's validity
    if is_key_valid:
        st.sidebar.success("✅ Gemini API key loaded and is valid!")
    else:
        # Display instructions and input box if key is not valid or missing
        st.sidebar.warning("⚠️ No valid Gemini API key found. Enter a key below.")
        st.sidebar.markdown(
            """
            **To get a key:**
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
            2. Create a new API key.
            
            **For this session:** Paste the key below.
            
            **For a permanent solution:** Create a `.env` file in the same directory as this app.py with the following line:
            `GEMINI_API_KEY='your_api_key_here'`
            """
        )
        
        gemini_api_key_input = st.sidebar.text_input(
            "🔑 Paste your key here:", 
            type="password", 
            key="gemini_api_key_input"
        )

        if gemini_api_key_input:
            os.environ['GEMINI_API_KEY'] = gemini_api_key_input
            st.rerun() # Rerun to validate the new key

    return is_key_valid


def render_sidebar(available_models):
    """Configures the sidebar with model selection and settings."""
    st.sidebar.markdown("## 🎯 Model & Settings")
    model_options = [f"{model['symbol']} - {model['company_name']} (R²: {model['r2_score']:.3f})"
                     for model in available_models]

    initial_symbol = st.session_state.selected_symbol
    initial_idx = next((i for i, model in enumerate(available_models) if model['symbol'] == initial_symbol), 0)

    selected_idx = st.sidebar.selectbox("📊 Choose Trained Model",
                                         range(len(model_options)),
                                         index=initial_idx,
                                         format_func=lambda x: model_options[x],
                                         help="Select a model trained on a specific stock.")
    
    # Update selected symbol in session state to trigger a full reload of data
    if st.session_state.selected_symbol != available_models[selected_idx]['symbol']:
        st.session_state.selected_symbol = available_models[selected_idx]['symbol']
        st.rerun()

    
    selected_model = available_models[selected_idx]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Prediction & Display")
    # Retrieve the future_days from session state
    future_days = st.session_state.get("future_days", 30)
    future_days_new = st.sidebar.slider("Days to Predict", 1, 90, future_days, help="Choose how many days into the future to predict.")
    
    if future_days_new != future_days:
        st.session_state.future_days = future_days_new
        st.rerun()


    show_technical_details = st.sidebar.checkbox("Show Technical Details", True, help="Display a detailed table of technical indicators.")
    show_charts = st.sidebar.checkbox("Show Charts", True, help="Display interactive charts for analysis.")

    # Call the API key check function
    enable_gemini = check_and_configure_gemini()
    
    # Checkbox state is now dependent on the validation function
    st.sidebar.markdown("---")
    enable_gemini = st.sidebar.checkbox("Enable AI Insights", value=enable_gemini, disabled=not enable_gemini, help="Toggle AI insights on or off. Requires a valid API key.")

    return selected_model, future_days, show_technical_details, show_charts, enable_gemini

@st.cache_data
def get_historical_data(symbol, period):
    """Cached function to fetch historical data from yfinance."""
    data = yf.download(symbol, period=period, interval="1d")
    return data

def render_historical_graph(symbol):
    """Renders a graph of historical stock prices with a selectable time window."""
    st.markdown("## 📈 Historical Price Chart")
    
    time_windows = {
        "7D": "7d",
        "1M": "1mo",
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "5Y": "5y"
    }

    selected_window = st.radio(
        "Select Time Window",
        list(time_windows.keys()),
        horizontal=True,
        index=0
    )
    
    try:
        data = get_historical_data(symbol, time_windows[selected_window])
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty and 'Close' in data.columns:
            data.dropna(subset=['Close'], inplace=True)
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            data.dropna(subset=['Close'], inplace=True)

            if not data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#2a5298', width=2)
                ))
                fig.update_layout(
                    title=f"Historical Price for {symbol} ({selected_window})",
                    yaxis_title="Price (₹)",
                    xaxis_title="Date",
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                    yaxis=dict(
                        title="Price (₹)",
                        tickformat=".2f"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid historical data found for the selected time window after cleaning.")
        else:
            st.warning("No historical data found for the selected time window.")
    except Exception as e:
        st.error(f"Error fetching or plotting historical data: {str(e)}")
        st.error(traceback.format_exc())


def render_metrics(current_price, recent_data):
    """Renders polished and relevant top-level market metrics."""
    st.markdown("## 📊 Current Market Status")
    
    col1, col2, col3 = st.columns(3)
    
    price_delta = ((current_price - recent_data['Close'].iloc[-2]) / recent_data['Close'].iloc[-2] * 100)
    price_color_class = "positive" if price_delta > 0 else "negative"
    price_border_class = "positive-border" if price_delta > 0 else "negative-border"
    price_emoji = "▲" if price_delta > 0 else "▼"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card {price_border_class}">
            <p class="metric-title">Current Price</p>
            <h2 class="metric-value">₹{current_price:.2f}</h2>
            <p class="metric-delta {price_color_class}">{price_emoji} {price_delta:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    rsi_value = recent_data['RSI'].iloc[-1]
    rsi_status = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
    rsi_color_class = "negative" if rsi_value > 70 else "positive" if rsi_value < 30 else ""
    rsi_border_class = "negative-border" if rsi_value > 70 else "positive-border" if rsi_value < 30 else "neutral-border"

    with col2:
        st.markdown(f"""
        <div class="metric-card {rsi_border_class}">
            <p class="metric-title">RSI</p>
            <h2 class="metric-value">{rsi_value:.1f}</h2>
            <p class="metric-delta {rsi_color_class}">{rsi_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    volume_value = recent_data['Volume'].iloc[-1]
    volume_change = (volume_value - recent_data['Volume'].iloc[-2]) / recent_data['Volume'].iloc[-2] * 100 if recent_data['Volume'].iloc[-2] != 0 else 0
    volume_color_class = "positive" if volume_change > 0 else "negative"
    volume_border_class = "positive-border" if volume_change > 0 else "negative-border"
    volume_emoji = "▲" if volume_change > 0 else "▼"

    with col3:
        st.markdown(f"""
        <div class="metric-card {volume_border_class}">
            <p class="metric-title">Volume</p>
            <h2 class="metric-value">{volume_value:,.0f}</h2>
            <p class="metric-delta {volume_color_class}">{volume_change:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

def render_predictions(predictions, current_price, future_days):
    """Renders the AI prediction cards."""
    st.markdown("## 🔮 AI Predictions")
    periods = {
        "1 Day": 1, 
        "1 Week": 7, 
        "2 Weeks": 14, 
        "1 Month": 30, 
        "3 Months": 90
    }
    
    periods_to_display = {
        name: days for name, days in periods.items() if days <= future_days
    }
    
    pred_cols = st.columns(len(periods_to_display))
    
    for idx, (period_name, days) in enumerate(periods_to_display.items()):
        gain_pct, status, pred_price = calculate_gain_loss(predictions, current_price, days)
        
        with pred_cols[idx]:
            color_class = "gain-positive" if gain_pct > 0 else "gain-negative"
            emoji = "📈" if gain_pct > 0 else "📉"
            text_color_class = "gain-text" if gain_pct > 0 else "loss-text"
            
            st.markdown(f"""
            <div class="prediction-card {color_class}">
                <h4>{period_name}</h4>
                <h2 class="{text_color_class}">{emoji} ₹{pred_price:.2f}</h2>
                <h3 class="{text_color_class}">{gain_pct:+.2f}%</h3>
                <p>{status}</p>
            </div>
            """, unsafe_allow_html=True)

def render_recommendation_gauge(rec_score):
    """Renders a speedometer-style gauge for the recommendation score."""
    st.markdown("## 🎯 AI Recommendation")
    
    # Map score to a category
    if rec_score >= 3:
        recommendation = "STRONG BUY"
        rec_class = "strong-buy-color"
        needle_color = "#11783f"
    elif rec_score >= 1:
        recommendation = "BUY"
        rec_class = "buy-color"
        needle_color = "#28a745"
    elif rec_score >= -1:
        recommendation = "HOLD"
        rec_class = "hold-color"
        needle_color = "#ffc107"
    elif rec_score >= -3:
        recommendation = "SELL"
        rec_class = "sell-color"
        needle_color = "#fd7e14"
    else:
        recommendation = "STRONG SELL"
        rec_class = "strong-sell-color"
        needle_color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = rec_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<span style='font-size:1.5em; font-weight:bold;' class='{rec_class}'>{recommendation}</span><br><span style='font-size:0.9em;'>Score: {rec_score:.1f}/5</span>", 'font': {'size': 20}},
        gauge = {
            'shape': "angular",
            'axis': {'range': [-5, 5], 'tickvals': [-5,-3,-1,1,3,5], 'ticktext': ['-5','-3','-1','1','3','5']},
            'bar': {'color': needle_color, 'thickness': 0.3},
            'bgcolor': "white",
            'bordercolor': "gray",
            'steps': [
                {'range': [-5, -3], 'color': '#dc3545'},
                {'range': [-3, -1], 'color': '#fd7e14'},
                {'range': [-1, 1], 'color': '#ffc107'},
                {'range': [1, 3], 'color': '#28a745'},
                {'range': [3, 5], 'color': '#11783f'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': rec_score}
        }
    ))
    fig.update_layout(height=400, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)


def render_recommendation(rec_score, rec_factors):
    """Renders the AI recommendation and analysis factors."""
    rec_col1, rec_col2 = st.columns([1, 2])

    with rec_col1:
        render_recommendation_gauge(rec_score)
    
    with rec_col2:
        st.markdown("### Analysis Factors:")
        st.markdown('<ul class="analysis-factors">', unsafe_allow_html=True)
        for factor in rec_factors:
            try:
                score_str = factor.split('(')[-1].replace(')', '')
                score = float(score_str.replace('+', ''))
                score_class = "score-positive" if score > 0 else "score-negative"
            except (IndexError, ValueError):
                score_class = ""
                score_str = ""
            
            factor_description = factor.split('(')[0].strip()
            
            st.markdown(f"""
            <li>
                <span>{factor_description}</span>
                <span class="score-pill {score_class}">{score_str}</span>
            </li>
            """, unsafe_allow_html=True)
        st.markdown('</ul>', unsafe_allow_html=True)

def render_charts(recent_data, predictions, future_days, symbol):
    """Renders interactive Plotly charts."""
    st.markdown("## 📈 Technical Analysis Charts")
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Historical prices and moving averages
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['Close'], name="Historical Price", line=dict(color="#2a5298")), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['EMA20'], name="EMA20", line=dict(color="#ffc107", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['EMA50'], name="EMA50", line=dict(color="#dc3545", dash="dash")), row=1, col=1)

    # Add a dotted line connecting historical and predicted data
    connector_dates = [recent_data.index[-1], pd.date_range(start=recent_data.index[-1] + timedelta(days=1), periods=1)[0]]
    connector_prices = [recent_data['Close'].iloc[-1], predictions[0]]
    fig.add_trace(go.Scatter(x=connector_dates, y=connector_prices, 
                             mode='lines', name="Prediction Start", 
                             line=dict(color="#28a745", width=2, dash="dash")), row=1, col=1)
    
    # Predictions
    future_dates = pd.date_range(start=recent_data.index[-1] + timedelta(days=1), periods=future_days, freq='D')
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, name="AI Prediction", line=dict(color="#28a745", width=3)), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['RSI'], name="RSI", line=dict(color="#667eea")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc3545", row=2, col=1, annotation_text="Overbought", annotation_position="top left")
    fig.add_hline(y=30, line_dash="dash", line_color="#28a745", row=2, col=1, annotation_text="Oversold", annotation_position="bottom left")
    
    # MACD
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD'], name="MACD", line=dict(color="#17a2b8")), row=3, col=1)
    fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['MACD_Signal'], name="MACD Signal", line=dict(color="#ffc107")), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True,
                      title_text=f"<b>{symbol} Technical Analysis & AI Predictions</b>")
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    fig.update_layout(template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)

def render_technical_table(recent_data, current_price):
    """Renders a detailed table of technical indicators with hover help text."""
    st.markdown("## 📊 Current Technical Indicators")
    features = ["RSI", "MACD", "MACD_Signal", "EMA20", "EMA50", "ATR", "Current Price"]
    # Define the help text in a dictionary for easy mapping
    # help_dict = {
    #     "RSI": "The Relative Strength Index measures momentum. A value above 70 suggests the stock is overbought, while a value below 30 suggests it is oversold.",
    #     "MACD": "The Moving Average Convergence Divergence identifies changes in a stock's trend by comparing two moving averages.",
    #     "MACD Signal": "The MACD signal line is a moving average of the MACD itself, used to generate buy/sell signals.",
    #     "EMA20": "The 20-day Exponential Moving Average smooths price data to identify a short-term trend.",
    #     "EMA50": "The 50-day Exponential Moving Average smooths price data to identify a long-term trend.",
    #     "ATR": "The Average True Range measures a stock's volatility. A higher value indicates larger price swings.",
    #     "Current Price": "The last traded price of the stock."
    # }

    tech_data = {
        "Indicator": features,
        "Value": [
            f"{recent_data['RSI'].iloc[-1]:.2f}",
            f"{recent_data['MACD'].iloc[-1]:.4f}",
            f"{recent_data['MACD_Signal'].iloc[-1]:.4f}",
            f"₹{recent_data['EMA20'].iloc[-1]:.2f}",
            f"₹{recent_data['EMA50'].iloc[-1]:.2f}",
            f"₹{recent_data['ATR'].iloc[-1]:.2f}",
            f"₹{current_price:.2f}"
        ],
        "Signal": [
            "Overbought" if recent_data['RSI'].iloc[-1] > 70 else "Oversold" if recent_data['RSI'].iloc[-1] < 30 else "Neutral",
            "Bullish" if recent_data['MACD'].iloc[-1] > recent_data['MACD_Signal'].iloc[-1] else "Bearish",
            "-",
            "Support" if current_price > recent_data['EMA20'].iloc[-1] else "Resistance",
            "Support" if current_price > recent_data['EMA50'].iloc[-1] else "Resistance",
            "-",
            "-"
        ]
    }
    tech_df = pd.DataFrame(tech_data)
    
    st.dataframe(
        tech_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Indicator": st.column_config.TextColumn(
                "Indicator"
                # help=help_dict
            )
        }
    )

def render_gemini_insights(selected_model, current_price, predictions, recent_data, metadata, rec_score, future_days):
    """Generates and renders insights using the Gemini API with a more detailed prompt."""
    st.markdown("## 🤖 AI Market Insights")
    
    if not GEMINI_AVAILABLE:
        st.warning("AI Insights are not available. Please install the Google Generative AI library.")
        return

    # Check if insights are already in the cache for the current stock
    insights_key = f"gemini_insights_{selected_model['symbol']}"
    if insights_key in st.session_state.insights_cache:
        st.markdown(st.session_state.insights_cache[insights_key])
        return
        
    with st.spinner("🧠 Generating comprehensive AI insights..."):
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

            # Load prompt template from prompts/gemini.txt
            prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../prompts/gemini.txt"))
            prompt_template = ""
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt_template = f.read()
                except Exception as e:
                    st.error(f"Error loading Gemini prompt template: {e}")
            if not prompt_template:
                # Fallback to default prompt
                prompt_template = (
                    "As a senior financial analyst, provide a comprehensive investment analysis for {company_name} ({symbol}).\n\n"
                    "Analyze the following data points to create a structured report. All numerical values should be formatted clearly.\n\n"
                    "**Provided Data:**\n"
                    "- **Current Price:** ₹{current_price:.2f}\n"
                    "- **Sector:** {sector}\n"
                    "- **AI Model Accuracy (R²):** {model_accuracy:.3f}\n"
                    "- **AI Price Predictions:**\n"
                    "  - 1 Week: {predictions_1_week_gain_pct:.1f}% gain\n"
                    "  - 1 Month: {predictions_1_month_gain_pct:.1f}% gain\n"
                    "- **Technical Indicators:**\n"
                    "  - RSI: {RSI:.1f}\n"
                    "  - MACD: {MACD:.4f}\n"
                    "  - MACD Signal: {MACD_Signal:.4f}\n"
                    "  - EMA20: {EMA20:.2f}\n"
                    "  - EMA50: {EMA50:.2f}\n"
                    "  - ATR: {ATR:.2f}\n"
                    "- **Recommendation Score:** {recommendation_score}/5\n\n"
                    "**Report Structure:**\n"
                    "1.  **Features meaning:** Explain the significance of the features RSI, MACD, MACD Signal, EMA20, EMA50, ATR, and Current Price.\n"
                    "2.  **Key Insights:** Summarize the main takeaways from the AI analysis and technical data.\n"
                    "3.  **Risk Factors:** Identify and explain the primary risks, including those not mentioned in the provided data, such as market risk or company-specific news.\n"
                    "4.  **Investment Outlook:** Based on the data, provide a clear short- to medium-term outlook.\n"
                    "5.  **Strategic Recommendations:** Offer actionable, strategic advice for a retail investor (e.g., 'Buy on Dips,' 'Hold,' 'Book Partial Profits').\n"
                    "6.  **Conclusion:** Deliver a final, concise summary of the overall investment thesis.\n\n"
                    "Ensure the entire output is formatted using Markdown with appropriate headings and bolding for clarity. Do not include any introductory or concluding remarks outside of the requested report structure."
                )

            # Build flat prompt_vars dict for formatting
            try:
                prompt_vars = {
                    "company_name": stock_data["company_name"],
                    "symbol": stock_data["symbol"],
                    "current_price": stock_data["current_price"],
                    "sector": stock_data["sector"],
                    "model_accuracy": stock_data["model_accuracy"],
                    "predictions_1_week_gain_pct": stock_data["predictions"]["1 Week"]["gain_pct"],
                    "predictions_1_month_gain_pct": stock_data["predictions"]["1 Month"]["gain_pct"],
                    "RSI": stock_data["technical_indicators"]["RSI"],
                    "MACD": stock_data["technical_indicators"]["MACD"],
                    "MACD_Signal": stock_data["technical_indicators"]["MACD_Signal"],
                    "EMA20": stock_data["technical_indicators"]["EMA20"],
                    "EMA50": stock_data["technical_indicators"]["EMA50"],
                    "ATR": recent_data["ATR"].iloc[-1],
                    "recommendation_score": stock_data["recommendation_score"]
                }
                prompt = prompt_template.format(**prompt_vars)
            except Exception as e:
                st.error(f"Error formatting Gemini prompt: {e}")
                prompt = ""
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            st.session_state.insights_cache[insights_key] = response.text
            st.markdown(response.text)

        except genai.types.BlockedPromptException as e:
            st.warning("⚠️ The AI model's response was blocked due to safety concerns. Please try again later.")
            st.exception(e)
        except Exception as e:
            if "quota" in str(e).lower():
                st.info("⚠️ Gemini API usage limit reached. Please try refreshing the page after a few minutes to get AI insights.")
            else:
                st.error(f"An unexpected API error occurred: {e}")
                st.exception(e)


def render_performance_section(info, current_price):
    """Renders the Performance section with 52-week high/low and other key metrics."""
    st.markdown("## 📈 Performance")
    
    try:
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
        day_low = info.get('dayLow', 'N/A')
        day_high = info.get('dayHigh', 'N/A')
        open_price = info.get('open', 'N/A')
        previous_close = info.get('previousClose', 'N/A')
        volume = info.get('volume', 'N/A')
        market_cap = info.get('marketCap', 'N/A')

        st.markdown("### Today's Range")
        range_col1, range_col2 = st.columns([1, 1])
        with range_col1:
            st.markdown(f"<b>Today's Low:</b> {day_low}", unsafe_allow_html=True)
        with range_col2:
            st.markdown(f'<div style="text-align: right;"><b>Today\'s High:</b> {day_high}</div>', unsafe_allow_html=True)
        st.slider("Today's Range", float(day_low) if isinstance(day_low, (int, float)) else 0, float(day_high) if isinstance(day_high, (int, float)) else 1000, float(current_price) if isinstance(current_price, (int, float)) else 0, disabled=True)
        st.write("---")

        st.markdown("### 52-Week Range")
        w_col1, w_col2 = st.columns([1, 1])
        with w_col1:
            st.markdown(f"<b>52W Low:</b> {fifty_two_week_low}", unsafe_allow_html=True)
        with w_col2:
            st.markdown(f'<div style="text-align: right;"><b>52W High:</b> {fifty_two_week_high}</div>', unsafe_allow_html=True)
        st.slider("52W Range", float(fifty_two_week_low) if isinstance(fifty_two_week_low, (int, float)) else 0, float(fifty_two_week_high) if isinstance(fifty_two_week_high, (int, float)) else 1000, float(current_price) if isinstance(current_price, (int, float)) else 0, disabled=True)
        st.write("---")

        metrics_data = {
            "Open": open_price,
            "Prev. Close": previous_close,
            "Volume": volume,
            "Market Cap": market_cap
        }

        metrics_df = pd.DataFrame(metrics_data, index=["Value"]).T
        st.dataframe(metrics_df, use_container_width=True)

    except Exception as e:
        st.error("⚠️ An error occurred while rendering the performance data.")
        st.info("The data for this stock might be incomplete or malformed. Please try another stock.")
        st.exception(e)
        st.error(traceback.format_exc())

# -------------------------
# Main Application Flow
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="MarketSage - AI Stock Predictor", page_icon="📈")
    
    load_dotenv(override=True)
    available_models = get_available_models()

    if not available_models:
        st.error("❌ No trained models found!")
        st.info("To get started, please train a model by running `python main.py` and then refresh this page.")
        st.stop()

    # --- Session State Management ---
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'
        st.session_state.selected_symbol = None
        st.session_state.future_days = 30
        st.session_state.insights_cache = {}

    if st.session_state.page == 'welcome' or st.session_state.selected_symbol is None:
        render_welcome_page(available_models)
        st.sidebar.markdown("---")
        st.sidebar.info("Select a stock on the main page to get started.")
        st.stop()
    
    # Check if a different stock was selected in the sidebar
    # This is handled by the sidebar function's rerun logic now.
    
    # --- Main App Content ---
    
    selected_model = next((model for model in available_models if model['symbol'] == st.session_state.selected_symbol), None)
    if not selected_model:
        st.error("Selected model not found. Please try refreshing.")
        st.session_state.page = 'welcome'
        st.rerun()

    
    try:
        selected_model, future_days, show_technical_details, show_charts, enable_gemini = render_sidebar(available_models)
    except Exception as e:
        st.error(f"Error in sidebar rendering: {e}")
        st.error(traceback.format_exc())
        st.stop()
        
    render_header(selected_model['symbol'], selected_model['company_name'])
    
    try:
        with st.spinner("🤖 Loading AI models and data..."):
            metadata, scaler = load_model_data(selected_model['path'])
            if metadata is None: st.stop()

            lstm_model, transformer_model, device = load_models(selected_model['path'], metadata)
            if lstm_model is None: st.stop()

            full_data, info = prepare_recent_data(selected_model['symbol'])
            if full_data is None: st.stop()
            
            # The fix is here: create features_for_model using metadata
            # Get the list of features from the metadata
            model_features_list = metadata['features']
            features_for_model = full_data[model_features_list].copy()

            predictions = predict_future(lstm_model, transformer_model, features_for_model, scaler, metadata, device, future_days)
            if predictions is None: st.stop()

        current_price = full_data['Close'].iloc[-1]
        rec_score, rec_factors = calculate_recommendation_score(full_data, metadata['model_metrics']['hybrid']['r2'], predictions, current_price)

        render_historical_graph(selected_model['symbol'])
        st.write("---")
        st.write("---")
        if info:
            render_performance_section(info, current_price)
        render_metrics(current_price, full_data)
        st.write("---")
        render_predictions(predictions, current_price, future_days)
        st.write("---")
        render_recommendation(rec_score, rec_factors)
        
        st.write("---")
        if show_charts:
            render_charts(full_data, predictions, future_days, selected_model['symbol'])
        
        if show_technical_details:
            st.write("---")
            render_technical_table(full_data, current_price)

        if enable_gemini and GEMINI_AVAILABLE:
            st.write("---")
            render_gemini_insights(selected_model, current_price, predictions, full_data, metadata, rec_score, future_days)

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
        st.error(traceback.format_exc())
    
    st.markdown("""
---
## 🛠️ Open Source Code

You can view, fork, or contribute to the full MarketSage project on GitHub:

[🔗 GitHub Repository: Lazy-Coder-03/MarketSage](https://github.com/lazy-coder-03/MarketSage)

Feel free to star ⭐ the repo, open issues, or submit pull requests!

""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📈 MarketSage - AI-Powered Stock Prediction System</p>
        <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()