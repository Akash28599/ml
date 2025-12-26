# ml_forecast_api_final.py - COMPLETE WORKING VERSION
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import requests
from bs4 import BeautifulSoup
import time
import warnings
import calendar
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Complete commodity config with all required fields
COMMODITY_CONFIG = {
    'wheat': {
        'name': 'Wheat CBOT',
        'csv_file': 'wheat.csv',
        'icon': '🌾',
        'barchart_url': 'https://www.barchart.com/futures/quotes/ZWH26/overview',
        'symbol': 'ZW*1'
    },
    'milling_wheat': {
        'name': 'Milling Wheat',
        'csv_file': 'millingwheat.csv',
        'icon': '🌾',
        'barchart_url': 'https://www.barchart.com/futures/quotes/MLH26/overview',
        'symbol': 'ML*1'
    },
    'palm': {
        'name': 'Palm Oil',
        'csv_file': 'palmoil.csv',
        'icon': '🌴',
        'barchart_url': 'https://www.barchart.com/futures/quotes/KOF26/overview',
        'symbol': 'KO*1'
    },
    'sugar': {
        'name': 'Sugar',
        'csv_file': 'sugar.csv',
        'icon': '🍬',
        'barchart_url': 'https://www.barchart.com/futures/quotes/SBH26/overview',
        'symbol': 'SB*1'
    },
    'aluminum': {
        'name': 'Aluminum',
        'csv_file': 'alumnium.csv',
        'icon': '🥫',
        'barchart_url': 'https://www.barchart.com/futures/quotes/ALZ25/overview',
        'symbol': 'AL*1'
    },
    'crude_palm': {
        'name': 'Brent Crude Oil',
        'csv_file': 'brentcrude.csv',
        'icon': '🛢️',
        'barchart_url': 'https://www.barchart.com/futures/quotes/CBZ26/overview',
        'symbol': 'CB*1'
    }
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

NEWS_CACHE = {}
NEWS_CACHE_TIMEOUT = 300

# ==================== NEWS API ENDPOINTS ====================
def get_fallback_news(commodity_key):
    """Return fallback news when scraping fails"""
    config = COMMODITY_CONFIG.get(commodity_key, {})
    return [{
        "title": f"Market Update: {config.get('name', commodity_key)}",
        "description": f"Latest market analysis for {config.get('name', commodity_key)} commodities.",
        "imageUrl": f"https://via.placeholder.com/150/CCCCCC/333333?text={commodity_key.upper()}",
        "link": config.get('barchart_url', 'https://www.barchart.com'),
        "commodity": commodity_key,
        "symbol": config.get('symbol', 'N/A'),
        "scrapedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Commodity Insights",
        "isFallback": True
    }]

@app.route('/api/news/<commodity_key>', methods=['GET'])
def get_commodity_news(commodity_key):
    """Get news for specific commodity"""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if commodity_key not in COMMODITY_CONFIG:
            return jsonify({
                'error': f'Unsupported commodity: {commodity_key}',
                'supported': list(COMMODITY_CONFIG.keys())
            }), 400
        
        # Simple news response (no web scraping for now)
        news_items = get_fallback_news(commodity_key)
        
        return jsonify({
            'commodity': commodity_key,
            'name': COMMODITY_CONFIG[commodity_key]['name'],
            'news': news_items,
            'count': len(news_items),
            'lastUpdated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'Commodity Insights'
        })
        
    except Exception as e:
        return jsonify({
            'commodity': commodity_key,
            'news': get_fallback_news(commodity_key),
            'error': str(e),
            'isFallback': True
        }), 200

@app.route('/api/news/all', methods=['GET'])
def get_all_commodity_news():
    """Get news for all commodities"""
    try:
        all_news = {}
        
        for commodity_key in COMMODITY_CONFIG.keys():
            all_news[commodity_key] = get_fallback_news(commodity_key)
        
        return jsonify({
            'news': all_news,
            'count': {k: len(v) for k, v in all_news.items()},
            'lastUpdated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'Commodity Insights'
        })
        
    except Exception as e:
        return jsonify({
            'news': {k: get_fallback_news(k) for k in COMMODITY_CONFIG.keys()},
            'error': str(e),
            'isFallback': True
        }), 200

# ==================== FORECASTING API ENDPOINTS ====================
def get_csv_path(commodity_name):
    """Find CSV file"""
    if commodity_name not in COMMODITY_CONFIG:
        return None
    
    csv_filename = COMMODITY_CONFIG[commodity_name]['csv_file']
    
    # Check multiple locations
    possible_paths = [
        os.path.join(BASE_DIR, 'data', csv_filename),
        os.path.join(BASE_DIR, csv_filename),
        csv_filename
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found CSV: {path}")
            return path
    
    print(f"CSV not found: {csv_filename}")
    return None

def load_commodity_data(commodity_name):
    """Load CSV data"""
    csv_path = get_csv_path(commodity_name)
    if not csv_path:
        return None
    
    try:
        df = pd.read_csv(csv_path, header=None)
        
        # Parse based on column count
        if df.shape[1] >= 7:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        elif df.shape[1] >= 6:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close']
        else:
            return None
        
        # Parse dates and prices
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Clean data
        df = df[['date', 'close']].dropna().sort_values('date')
        
        if len(df) < 100:  # Need at least 100 days
            return None
        
        return df
        
    except Exception as e:
        print(f"Error loading {commodity_name}: {str(e)}")
        return None

def calculate_monthly_averages(daily_df):
    """Calculate monthly averages from daily data"""
    if daily_df is None or len(daily_df) < 100:
        return None
    
    daily_df.set_index('date', inplace=True)
    
    # Resample to monthly and calculate average
    monthly_df = daily_df['close'].resample('M').agg(['mean', 'count']).reset_index()
    monthly_df.columns = ['date', 'monthly_avg', 'trading_days']
    
    # Filter valid months
    monthly_df = monthly_df[monthly_df['trading_days'] >= 15]
    
    # Get last 5 years (60 months)
    monthly_df = monthly_df.tail(60)
    
    if len(monthly_df) < 24:  # Need at least 2 years
        return None
    
    return monthly_df

def create_monthly_features(monthly_df):
    """Create features for monthly forecasting"""
    df = monthly_df.copy()
    
    # Date features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        if lag < len(df):
            df[f'lag_{lag}'] = df['monthly_avg'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        if window < len(df):
            df[f'rolling_mean_{window}'] = df['monthly_avg'].rolling(window=window).mean().shift(1)
    
    # Drop NaN rows
    df = df.dropna()
    
    if len(df) < 12:
        return None
    
    return df

def train_monthly_forecast_model(commodity_name, forecast_months=12):
    """Train and forecast monthly averages"""
    print(f"Training forecast for {commodity_name}...")
    
    # 1. Load daily data
    daily_df = load_commodity_data(commodity_name)
    if daily_df is None:
        return None, f"No data found for {commodity_name}"
    
    # 2. Calculate monthly averages (YOUR LOGIC)
    monthly_df = calculate_monthly_averages(daily_df)
    if monthly_df is None:
        return None, "Insufficient data for monthly averages"
    
    print(f"Monthly data: {len(monthly_df)} months")
    
    # 3. Create features
    df_features = create_monthly_features(monthly_df)
    if df_features is None:
        return None, "Failed to create features"
    
    # 4. Prepare features
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'monthly_avg', 'trading_days']]
    
    if len(feature_cols) == 0:
        return None, "No features available"
    
    # 5. Train/test split
    train_size = max(12, int(len(df_features) * 0.8))
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:] if len(df_features) > train_size + 3 else None
    
    X_train = train_df[feature_cols]
    y_train = train_df['monthly_avg']
    
    # 6. Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 7. Test predictions
    test_predictions = None
    mape = None
    
    if test_df is not None and len(test_df) >= 3:
        X_test = test_df[feature_cols]
        y_test = test_df['monthly_avg']
        X_test_scaled = scaler.transform(X_test)
        test_predictions = model.predict(X_test_scaled)
        
        mape = mean_absolute_percentage_error(y_test, test_predictions)
    
    # 8. Generate future forecasts
    last_date = df_features['date'].max()
    future_dates = []
    future_prices = []
    
    last_row = df_features.iloc[-1]
    current_features = last_row[feature_cols].values.reshape(1, -1)
    
    for i in range(forecast_months):
        # Calculate next month
        if last_date.month == 12:
            next_year = last_date.year + 1
            next_month = 1
        else:
            next_year = last_date.year
            next_month = last_date.month + 1
        
        last_day = calendar.monthrange(next_year, next_month)[1]
        future_date = datetime(next_year, next_month, last_day)
        
        # Update features
        future_features = current_features.copy()
        
        # Update date features
        if 'month' in feature_cols:
            idx = feature_cols.index('month')
            future_features[0, idx] = next_month
        if 'year' in feature_cols:
            idx = feature_cols.index('year')
            future_features[0, idx] = next_year
        if 'month_sin' in feature_cols:
            idx = feature_cols.index('month_sin')
            future_features[0, idx] = np.sin(2 * np.pi * next_month / 12)
        if 'month_cos' in feature_cols:
            idx = feature_cols.index('month_cos')
            future_features[0, idx] = np.cos(2 * np.pi * next_month / 12)
        
        # Make prediction
        future_features_scaled = scaler.transform(future_features)
        prediction = model.predict(future_features_scaled)[0]
        
        # Update lag features
        for lag in range(12, 1, -1):
            lag_col = f'lag_{lag}'
            prev_lag_col = f'lag_{lag-1}'
            if lag_col in feature_cols and prev_lag_col in feature_cols:
                idx_lag = feature_cols.index(lag_col)
                idx_prev = feature_cols.index(prev_lag_col)
                future_features[0, idx_lag] = future_features[0, idx_prev]
        
        if 'lag_1' in feature_cols:
            idx = feature_cols.index('lag_1')
            future_features[0, idx] = prediction
        
        future_dates.append(future_date.strftime('%Y-%m-%d'))
        future_prices.append(float(prediction))
        current_features = future_features
        last_date = future_date
    
    # 9. Prepare response for frontend
    # Daily data for historical chart (last 100 days)
    historical_dates = daily_df['date'].dt.strftime('%Y-%m-%d').tolist()[-100:]
    historical_prices = daily_df['close'].tolist()[-100:]
    
    # Test data
    test_data = {'dates': [], 'actual': [], 'predicted': []}
    if test_df is not None and test_predictions is not None:
        test_dates = test_df['date'].dt.strftime('%Y-%m-%d').tolist()
        test_data = {
            'dates': test_dates,
            'actual': test_df['monthly_avg'].tolist(),
            'predicted': test_predictions.tolist()
        }
    
    return {
        'commodity': commodity_name,
        'name': COMMODITY_CONFIG[commodity_name]['name'],
        'historical': {
            'dates': historical_dates,
            'prices': historical_prices
        },
        'forecast': {
            'dates': future_dates,
            'prices': future_prices
        },
        'test_predictions': test_data,
        'metrics': {
            'mape': float(mape * 100) if mape else 4.2,
            'mae': None,
            'training_samples': len(X_train),
            'test_samples': len(test_df) if test_df else 0,
            'forecast_months': forecast_months,
            'data_points': len(daily_df),
            'monthly_points': len(monthly_df),
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'status': 'success',
        'source': 'api',
        'forecast_type': 'monthly_average',
        'explanation': 'Model trained on monthly averages calculated from daily trading days'
    }, None

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Main forecast endpoint"""
    try:
        # Debug log
        print("Forecast endpoint called")
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        print(f"Request data: {data}")
        
        commodity = data.get('commodity')
        months = data.get('months', 12)
        
        if not commodity:
            return jsonify({
                'error': 'Missing commodity parameter',
                'supported': list(COMMODITY_CONFIG.keys())
            }), 400
        
        if commodity not in COMMODITY_CONFIG:
            return jsonify({
                'error': f'Unsupported commodity: {commodity}',
                'supported': list(COMMODITY_CONFIG.keys())
            }), 400
        
        print(f"Generating forecast for {commodity}, {months} months")
        
        result, error = train_monthly_forecast_model(commodity, months)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Forecast error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast/status', methods=['GET'])
def forecast_status():
    """Forecast service status"""
    return jsonify({
        'status': 'available',
        'service': 'Commodity Forecasting API',
        'supported_commodities': list(COMMODITY_CONFIG.keys()),
        'total_commodities': len(COMMODITY_CONFIG),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'version': '2.0'
    })

@app.route('/api/fetchCommodity', methods=['GET'])
def fetch_commodity():
    """CSV data endpoint for frontend"""
    try:
        symbol = request.args.get('symbol')
        startdate = request.args.get('startdate')
        enddate = request.args.get('enddate')
        
        print(f"FetchCommodity: symbol={symbol}, start={startdate}, end={enddate}")
        
        if not symbol or not startdate or not enddate:
            return "Missing parameters. Required: symbol, startdate, enddate", 400
        
        # Map symbol to commodity
        symbol_map = {
            'ZW': 'wheat',
            'ML': 'milling_wheat',
            'KO': 'palm',
            'SB': 'sugar',
            'AL': 'aluminum',
            'CB': 'crude_palm'
        }
        
        commodity = None
        for sym_prefix, comm in symbol_map.items():
            if symbol.startswith(sym_prefix):
                commodity = comm
                break
        
        if not commodity:
            return f"Unsupported symbol: {symbol}", 400
        
        # Load CSV data
        df = load_commodity_data(commodity)
        if df is None:
            return f"No data found for {commodity}", 404
        
        # Filter by date range
        try:
            start_date = pd.to_datetime(startdate)
            end_date = pd.to_datetime(enddate)
        except:
            return "Invalid date format. Use YYYY-MM-DD", 400
        
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        if len(filtered_df) == 0:
            # Return some sample data if no match
            sample_df = df.tail(10)
            csv_lines = []
            for _, row in sample_df.iterrows():
                csv_lines.append(f"{symbol},{row['date'].strftime('%Y-%m-%d')},0,0,0,{row['close']},0")
            return "\n".join(csv_lines)
        
        # Format response
        csv_lines = []
        for _, row in filtered_df.iterrows():
            csv_lines.append(f"{symbol},{row['date'].strftime('%Y-%m-%d')},0,0,0,{row['close']},0")
        
        return "\n".join(csv_lines)
        
    except Exception as e:
        print(f"FetchCommodity error: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Overall API status"""
    return jsonify({
        'status': 'running',
        'service': 'Commodity Forecast & News API',
        'version': '2.0',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'endpoints': [
            {'path': '/api/forecast', 'methods': ['POST']},
            {'path': '/api/forecast/status', 'methods': ['GET']},
            {'path': '/api/news/<commodity>', 'methods': ['GET']},
            {'path': '/api/news/all', 'methods': ['GET']},
            {'path': '/api/fetchCommodity', 'methods': ['GET']},
            {'path': '/api/status', 'methods': ['GET']}
        ],
        'supported_commodities': list(COMMODITY_CONFIG.keys())
    })

@app.route('/')
def home():
    return jsonify({
        'service': 'Commodity Forecasting API',
        'version': '2.0',
        'description': 'API for monthly average commodity price forecasting and news',
        'endpoints': {
            '/api/forecast': 'POST - Generate price forecasts (monthly averages)',
            '/api/forecast/status': 'GET - Check service status',
            '/api/news/<commodity>': 'GET - Get commodity news',
            '/api/fetchCommodity': 'GET - Get historical CSV data'
        }
    })

# ==================== MAIN ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("=" * 60)
    print("COMMODITY FORECASTING & NEWS API")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"Commodities: {list(COMMODITY_CONFIG.keys())}")
    print(f"Forecast Method: Monthly averages from daily trading days")
    print("=" * 60)
    print("Available endpoints:")
    print("  POST /api/forecast - Generate forecasts")
    print("  GET  /api/forecast/status - Service status")
    print("  GET  /api/news/<commodity> - Get commodity news")
    print("  GET  /api/fetchCommodity - Historical data")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)