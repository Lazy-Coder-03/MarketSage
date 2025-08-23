# MarketSage: An AI-Powered Stock Prediction System

MarketSage is an AI-driven application that predicts future stock prices using a hybrid model combining **Long Short-Term Memory (LSTM)** and **Transformer** neural networks. The system is designed to provide comprehensive investment analysis and actionable insights for retail investors, all presented through a polished web interface built with Streamlit.

-----

## 🚀 Key Features

  * **Hybrid Model:** Leverages the strengths of both LSTM (for sequential data) and Transformer (for attention mechanisms) models to provide robust price predictions.
  * **Comprehensive Analysis:** Generates a detailed report including key insights, risk factors, technical indicators (RSI, MACD, EMAs), and a clear investment outlook.
  * **AI-Powered Insights:** Integrates with the Google Gemini API to produce a narrative, professional-grade financial analysis.
  * **Intuitive UI:** A clean and interactive dashboard built with Streamlit allows users to select a stock, view historical data, and see future predictions with custom charts and metrics.
  * **Local Execution:** All models are trained and run locally, ensuring data privacy and offline functionality.

-----

## 📁 Directory Structure

```
MarketSage/
├── src/
│   ├── app.py
│   └── main.py
├── notebooks/
│   ├── HYBRID_LSTM+TRANSFORMER.ipynb
│   ├── HYBRID_LSTM+TRANSFORMER (1).ipynb
│   └── Marketsage.ipynb
├── saved_models/
│   ├── TCS.NS/
│   │   ├── metadata.json
│   │   ├── lstm_model.h5
│   │   ├── transformer_model.pth
│   │   └── scaler.pkl
│   ├── RELIANCE.NS/
│   │   └── ...
│   └── ...
├── .env.example
└── README.md (This file)
```

  * `src/`: Contains the main application and training scripts (`app.py` for Streamlit dashboard, `main.py` for model training).
  * `notebooks/`: Contains Jupyter notebooks for exploratory data analysis, model prototyping, and testing.
  * `saved_models/`: The primary directory for storing all trained models and their associated data (scalers, metadata, etc.). Each stock symbol has its own subdirectory.
  * `.env.example`: A template for environment variables, specifically for your Gemini API key.

-----

## ⚙️ How to Run Locally

Follow these steps to set up and run the MarketSage application on your local machine.

### Step 1: Clone the Repository

Clone the project from GitHub and navigate into the main directory.

```bash
git clone https://github.com/lazy-coder-03/MarketSage.git
cd MarketSage
```

### Step 2: Install Dependencies

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

*(**Note:** A `requirements.txt` file is required for this step. If it doesn't exist, you can generate one from the provided code by listing the libraries you see: `streamlit`, `yfinance`, `pandas`, `numpy`, `torch`, `scikit-learn`, `keras`, `plotly`, `python-dotenv`, `google-generativeai`.)*

### Step 3: Train the Models

Before you can run the web app, you need to train a model for a specific stock. The `main.py` script is set up to train a model for **RELIANCE.NS** by default.

```bash
python src/main.py
```

This script will:

1.  Download historical data for `RELIANCE.NS`.
2.  Preprocess the data and add technical indicators.
3.  Train both an LSTM and a Transformer model.
4.  Find the optimal weight for the hybrid model.
5.  Save the trained models, scaler, and metadata to the `saved_models/RELIANCE.NS` directory.

You can change the `SYMBOL` variable in `main.py` to train models for other stocks, such as `"TCS.NS"` or `"SBIN.NS"`.

### Step 4: Configure the Gemini API (Optional but Recommended)

For the "AI Market Insights" section to work, you need a Google Gemini API key.

1.  Create a copy of the `.env.example` file and rename it to `.env`.
2.  Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3.  Add your key to the `.env` file.

<!-- end list -->

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### Step 5: Run the Web Application

With the models trained and the API key configured, you can now launch the Streamlit app.

```bash
streamlit run src/app.py
```

This will start a local web server and open the MarketSage dashboard in your default web browser. You can then select the stock model you just trained from the sidebar.

-----

## 🤝 Contribution & License

This project is for educational and research purposes. Feel free to fork the repository, modify the models, and contribute to the project.

This project is built and maintained by [Sayantan Ghosh](https://github.com/lazy-coder-03).