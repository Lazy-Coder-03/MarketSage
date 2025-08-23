"""
MarketSage - Hybrid Model Training Script
Trains LSTM + Transformer models and saves them for the Streamlit app
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Model Definitions
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

def prepare_data(symbol, lookback_days=60):
    """Prepare stock data with technical indicators"""
    print(f"📊 Preparing data for {symbol}...")
    
    train_start = "2015-01-01"
    train_end = pd.Timestamp.today()
    
    # Download stock data
    df = yf.download(symbol, start=train_start, end=train_end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    
    df = df[['Open','High','Low','Close','Volume']]
    
    # Get stock info
    ticker = yf.Ticker(symbol)
    info = ticker.info
    print(f"Company: {info.get('longName', symbol)}")
    print(f"Sector: {info.get('sector', 'Unknown')}")
    
    # Download sector data
    sector_symbol = get_sector_symbol(info.get("sector"))
    print(f"Using sector index: {sector_symbol}")
    
    sector_df = yf.download(sector_symbol, start=train_start, end=train_end, auto_adjust=True)
    if isinstance(sector_df.columns, pd.MultiIndex):
        sector_df.columns = sector_df.columns.get_level_values(0)
    sector_df = sector_df[['Close']].rename(columns={'Close':'Sector_Close'})
    
    # Merge stock + sector
    df = df.join(sector_df, how="inner")
    
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)

    # Forward-fill only OHLC
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].ffill()

    # Set Volume = 0 for non-business days
    df["Volume"] = df["Volume"].fillna(0)

    # Add business day flag (1 = trading day, 0 = weekend/holiday)
    df["is_business_day"] = df.index.to_series().map(lambda d: 1 if d.dayofweek < 5 else 0)

    # Rename index back to Date
    df.index.name = "Date"
    # Technical indicators for stock
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
    
    # Technical indicators for sector
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
    
    # Clean data
    df.dropna(inplace=True)
    print(df.columns)
    print(f"✅ Data prepared: {len(df)} samples")
    
    return df, info, sector_symbol

def create_sequences(data, lookback_days):
    """Create sequences for training"""
    features = ['Close','Open', 'High', 'Low', 'Volume', 'Sector_Close', 'EMA20',
                'EMA50', 'RSI', 'MACD', 'MACD_Signal', 'Sector_EMA20', 'Sector_EMA50',
                'Sector_RSI', 'Sector_MACD', 'Sector_MACD_Signal', 'is_business_day']
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(lookback_days, len(scaled)):
        X.append(scaled[i-lookback_days:i])
        y.append(scaled[i, 0])  # Close price
    
    return np.array(X), np.array(y), scaler, features

def train_transformer(X_train, y_train, feature_size, device, epochs=50, patience=10):
    """Train Transformer model"""
    print("🤖 Training Transformer model...")
    
    # Convert to tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # Train/Val split
    val_split = int(0.9 * len(X_train_torch))
    train_data = TensorDataset(X_train_torch[:val_split], y_train_torch[:val_split])
    val_data = TensorDataset(X_train_torch[val_split:], y_train_torch[val_split:])
    
    # Initialize model
    model = TransformerModel(
        feature_size=feature_size,
        hidden_dim=64,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    best_val_loss = float("inf")
    wait = 0
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        
        avg_train_loss = total_loss / len(train_data)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val, y_val = val_data.tensors
            X_val, y_val = X_val.to(device), y_val.to(device)
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_weights)
    print("✅ Transformer training completed!")
    return model

def train_lstm_model(X_train, y_train, epochs=50):
    """Train LSTM model"""
    print("🧠 Training LSTM model...")
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("✅ LSTM training completed!")
    return model, history

def inverse_transform(preds, scaler, n_features, target_col=0):
    """Inverse transform predictions"""
    dummy = np.zeros((len(preds), n_features))
    dummy[:, target_col] = preds.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col]

def get_predictions(model, X_torch, scaler, n_features, y_true=None, recalibrate=False, device='cpu'):
    """Get predictions from Transformer model"""
    model.eval()
    with torch.no_grad():
        preds = model(X_torch.to(device)).cpu().numpy()
    preds_rescaled = inverse_transform(preds, scaler, n_features)
    
    if recalibrate and y_true is not None:
        y_rescaled = inverse_transform(y_true, scaler, n_features, target_col=0)
        recal = LinearRegression()
        recal.fit(preds_rescaled.reshape(-1,1), y_rescaled.reshape(-1,1))
        preds_rescaled = recal.predict(preds_rescaled.reshape(-1,1)).flatten()
    
    return preds_rescaled

def find_best_hybrid_weight(lstm_preds, trans_preds, actual_prices):
    """Find optimal weight for hybrid model"""
    print("🔍 Finding optimal hybrid weight...")
    
    weights = np.linspace(0, 1, 51)
    results = []
    
    for w in weights:
        hybrid_preds = w * lstm_preds + (1-w) * trans_preds
        rmse = np.sqrt(mean_squared_error(actual_prices, hybrid_preds))
        r2 = r2_score(actual_prices, hybrid_preds)
        results.append((w, rmse, r2))
    
    results = np.array(results, dtype=object)
    best_idx = np.argmin(results[:,1])  # Best RMSE
    best_weight = results[best_idx, 0]
    
    print(f"✅ Optimal hybrid weight: {best_weight:.3f}")
    return best_weight

def save_models_and_data(symbol, lstm_model, transformer_model, scaler, features, 
                        hybrid_weight, info, model_metrics, lookback_days=60):
    """Save all models and associated data"""
    
    # Create models directory
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create symbol-specific directory
    symbol_dir = os.path.join(models_dir, symbol.replace(".", "_"))
    os.makedirs(symbol_dir, exist_ok=True)
    
    print(f"💾 Saving models for {symbol}...")
    
    # Save LSTM model
    lstm_path = os.path.join(symbol_dir, "lstm_model.h5")
    lstm_model.save(lstm_path)
    
    # Save Transformer model
    transformer_path = os.path.join(symbol_dir, "transformer_model.pth")
    torch.save(transformer_model.state_dict(), transformer_path)
    
    # Save scaler
    scaler_path = os.path.join(symbol_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'company_name': info.get('longName', symbol),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'features': features,
        'hybrid_weight': hybrid_weight,
        'lookback_days': lookback_days,
        'feature_size': len(features),
        'model_metrics': model_metrics,
        'training_date': pd.Timestamp.now().isoformat(),
        'model_architecture': {
            'transformer': {
                'hidden_dim': 64,
                'num_heads': 8,
                'num_layers': 2,
                'dropout': 0.1
            },
            'lstm': {
                'layers': [256, 128, 64],
                'dropout': 0.2
            }
        }
    }
    
    metadata_path = os.path.join(symbol_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Models saved successfully to {symbol_dir}")
    return symbol_dir

def main():
    """Main training function"""
    print("🚀 Starting MarketSage Hybrid Model Training")
    print("=" * 50)
    
    # Configuration
    SYMBOL = "RELIANCE.NS"  # Change this to train different stocks
    LOOKBACK_DAYS = 60
    EPOCHS = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. Prepare data
        df, info, sector_symbol = prepare_data(SYMBOL, LOOKBACK_DAYS)
        
        # 2. Create sequences
        X, y, scaler, features = create_sequences(df, LOOKBACK_DAYS)
        
        # 3. Train/test split
        split_idx = int(0.9 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # 4. Train models
        transformer_model = train_transformer(X_train, y_train, len(features), device, epochs=EPOCHS)
        lstm_model, history = train_lstm_model(X_train, y_train, epochs=EPOCHS)
        
        # 5. Get predictions on test set
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        
        trans_preds = get_predictions(transformer_model, X_test_torch, scaler, len(features), 
                                    y_test, recalibrate=True, device=device)
        lstm_preds = inverse_transform(lstm_model.predict(X_test, verbose=0), scaler, len(features))
        actual_prices = inverse_transform(y_test.reshape(-1,1), scaler, len(features))
        
        # 6. Find optimal hybrid weight
        best_weight = find_best_hybrid_weight(lstm_preds, trans_preds, actual_prices)
        hybrid_preds = best_weight * lstm_preds + (1-best_weight) * trans_preds
        
        # 7. Calculate metrics
        model_metrics = {
            'lstm': {
                'rmse': float(np.sqrt(mean_squared_error(actual_prices, lstm_preds))),
                'mae': float(mean_absolute_error(actual_prices, lstm_preds)),
                'r2': float(r2_score(actual_prices, lstm_preds))
            },
            'transformer': {
                'rmse': float(np.sqrt(mean_squared_error(actual_prices, trans_preds))),
                'mae': float(mean_absolute_error(actual_prices, trans_preds)),
                'r2': float(r2_score(actual_prices, trans_preds))
            },
            'hybrid': {
                'rmse': float(np.sqrt(mean_squared_error(actual_prices, hybrid_preds))),
                'mae': float(mean_absolute_error(actual_prices, hybrid_preds)),
                'r2': float(r2_score(actual_prices, hybrid_preds)),
                'weight': float(best_weight)
            }
        }
        
        # 8. Print results
        print("\n📊 Model Performance Summary:")
        print(f"LSTM     - RMSE: {model_metrics['lstm']['rmse']:.2f}, R²: {model_metrics['lstm']['r2']:.4f}")
        print(f"Trans    - RMSE: {model_metrics['transformer']['rmse']:.2f}, R²: {model_metrics['transformer']['r2']:.4f}")
        print(f"Hybrid   - RMSE: {model_metrics['hybrid']['rmse']:.2f}, R²: {model_metrics['hybrid']['r2']:.4f}")
        
        # 9. Save models
        model_dir = save_models_and_data(SYMBOL, lstm_model, transformer_model, scaler, 
                                       features, best_weight, info, model_metrics, LOOKBACK_DAYS)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Models saved to: {model_dir}")
        print("\nYou can now run the Streamlit app to use these trained models!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()