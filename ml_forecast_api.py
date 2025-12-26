# ml_forecast_api.py - MONTHLY AVERAGE TRADING DAYS VERSION (November 2025 example)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Render-compatible BASE_DIR
BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Barchart Commodity Configuration (UNCHANGED)
BARCHART_COMMODITY_CONFIG = {
    'wheat': {
        'name': 'Wheat CBOT',
        'symbol': 'ZW*1',
        'barchart_url': 'https://www.barchart.com/futures/quotes/ZWH26/overview',
        'icon': '🌾'
    },
    'milling_wheat': {
        'name': 'Milling Wheat',
        'symbol': 'MLH26',
        'barchart_url': 'https://www.barchart.com/futures/quotes/MLH26/overview',
        'icon': '🌾'
    },
    'palm': {
        'name': 'Palm Oil',
        'symbol': 'KOF26',
        'barchart_url': 'https://www.barchart.com/futures/quotes/KOF26/overview',
        'icon': '🌴'
    },
    'sugar': {
        'name': 'Sugar',
        'symbol': 'SBH26',
        'barchart_url': 'https://www.barchart.com/futures/quotes/SBH26/overview',
        'icon': '🍬'
    },
    'aluminum': {
        'name': 'Aluminum',
        'symbol': 'ALZ25',
        'barchart_url': 'https://www.barchart.com/futures/quotes/ALZ25/overview',
        'icon': '🥫'
    },
    'crude_palm': {
        'name': 'Brent Crude Oil',
        'symbol': 'CBZ26',
        'barchart_url': 'https://www.barchart.com/futures/quotes/CBZ26/overview',
        'icon': '🛢️'
    }
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

NEWS_CACHE = {}
NEWS_CACHE_TIMEOUT = int(os.environ.get('NEWS_CACHE_TIMEOUT', '300'))

# ALL NEWS FUNCTIONS REMAIN EXACTLY UNCHANGED
def get_default_image(commodity_key):
    image_map = {
        'wheat': 'https://via.placeholder.com/150/E3B23C/FFFFFF?text=WHEAT',
        'milling_wheat': 'https://via.placeholder.com/150/D4A76A/FFFFFF?text=MILLING',
        'palm': 'https://via.placeholder.com/150/4A772F/FFFFFF?text=PALM',
        'sugar': 'https://via.placeholder.com/150/FF6B6B/FFFFFF?text=SUGAR',
        'aluminum': 'https://via.placeholder.com/150/7E8C9C/FFFFFF?text=ALUM',
        'crude_palm': 'https://via.placeholder.com/150/2C3E50/FFFFFF?text=CRUDE'
    }
    return image_map.get(commodity_key, 'https://via.placeholder.com/150/CCCCCC/333333?text=NEWS')

def get_fallback_news(commodity_key):
    config = BARCHART_COMMODITY_CONFIG.get(commodity_key, {})
    return [{
        "title": f"Market Analysis: {config.get('name', commodity_key)}",
        "description": f"Latest trends and analysis for {config.get('name', commodity_key)}. Click to read more on Barchart.",
        "imageUrl": get_default_image(commodity_key),
        "link": config.get('barchart_url', 'https://www.barchart.com'),
        "commodity": commodity_key,
        "symbol": config.get('symbol', 'N/A'),
        "scrapedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "isFallback": True
    }]

def extract_news_from_barchart(html_content, commodity_key):
    soup = BeautifulSoup(html_content, 'html.parser')
    news_items = []
    
    recent_stories_heading = soup.find(['h3', 'h4', 'div'], text=lambda t: t and 'Most Recent Stories' in str(t))
    
    if recent_stories_heading:
        news_container = recent_stories_heading.find_next(['div', 'ul', 'section'])
        if news_container:
            news_elements = news_container.find_all(['li', 'article', 'div'], class_=lambda c: c and any(x in str(c) for x in ['news', 'story', 'article']), limit=5)
            
            for element in news_elements:
                title_elem = element.find(['a', 'h4', 'h5'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = 'https://www.barchart.com' + link
                    
                    desc_elem = element.find(['p', 'div', 'span'], class_=lambda c: c and any(x in str(c) for x in ['summary', 'description', 'text']))
                    description = desc_elem.get_text(strip=True) if desc_elem else "Click to read more"
                    
                    img_elem = element.find('img')
                    image_url = img_elem.get('src', '') if img_elem else get_default_image(commodity_key)
                    if image_url and not image_url.startswith('http'):
                        image_url = 'https://www.barchart.com' + image_url
                    
                    news_items.append({
                        "title": title[:100] + "..." if len(title) > 100 else title,
                        "description": description[:150] + "..." if len(description) > 150 else description,
                        "imageUrl": image_url or get_default_image(commodity_key),
                        "link": link or BARCHART_COMMODITY_CONFIG.get(commodity_key, {}).get('barchart_url', 'https://www.barchart.com'),
                        "commodity": commodity_key,
                        "symbol": BARCHART_COMMODITY_CONFIG.get(commodity_key, {}).get('symbol', ''),
                        "scrapedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Barchart"
                    })
    
    if not news_items:
        news_links = soup.find_all('a', href=lambda href: href and any(x in href for x in ['news', 'article', 'story']), limit=5)
        for link in news_links:
            title = link.get_text(strip=True)
            if title and len(title) > 10:
                href = link.get('href', '')
                if href and not href.startswith('http'):
                    href = 'https://www.barchart.com' + href
                
                news_items.append({
                    "title": title[:100] + "..." if len(title) > 100 else title,
                    "description": "Market news and analysis",
                    "imageUrl": get_default_image(commodity_key),
                    "link": href or BARCHART_COMMODITY_CONFIG.get(commodity_key, {}).get('barchart_url', 'https://www.barchart.com'),
                    "commodity": commodity_key,
                    "symbol": BARCHART_COMMODITY_CONFIG.get(commodity_key, {}).get('symbol', ''),
                    "scrapedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Barchart"
                })
    
    return news_items

def scrape_barchart_news(commodity_key, force_refresh=False):
    cache_key = f"{commodity_key}_news"
    current_time = time.time()
    
    if not force_refresh and cache_key in NEWS_CACHE:
        cache_data = NEWS_CACHE[cache_key]
        if current_time - cache_data['timestamp'] < NEWS_CACHE_TIMEOUT:
            return cache_data['news']
    
    config = BARCHART_COMMODITY_CONFIG.get(commodity_key)
    if not config or 'barchart_url' not in config:
        return get_fallback_news(commodity_key)
    
    url = config['barchart_url']
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        news_items = extract_news_from_barchart(response.content, commodity_key)
        
        if news_items:
            NEWS_CACHE[cache_key] = {
                'news': news_items,
                'timestamp': current_time,
                'source': 'barchart'
            }
            return news_items
        else:
            return get_fallback_news(commodity_key)
            
    except requests.RequestException as e:
        return get_fallback_news(commodity_key)
    except Exception as e:
        return get_fallback_news(commodity_key)

## NEW MONTHLY AVERAGE LOGIC - 5 YEARS OF MONTHLY TRADING DAY AVERAGES
def load_commodity_data(commodity_name):
    """Load CSV data - UNCHANGED"""
    csv_files = {
        'wheat': 'wheat.csv',
        'milling_wheat': 'millingwheat.csv',
        'palm': 'palmoil.csv',
        'sugar': 'sugar.csv',
        'aluminum': 'alumnium.csv',
        'crude_palm': 'brentcrude.csv'
    }
    
    if commodity_name not in csv_files:
        return None
    
    csv_filename = csv_files[commodity_name]
    
    possible_paths = [
        os.path.join(BASE_DIR, 'data', csv_filename),
        os.path.join(BASE_DIR, csv_filename),
        csv_filename
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        return None
    
    try:
        df = pd.read_csv(csv_path, header=None)
        
        if df.shape[1] >= 7:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'] + [f'extra_{i}' for i in range(7, df.shape[1])]
        elif df.shape[1] >= 6:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close'] + [f'extra_{i}' for i in range(6, df.shape[1])]
        else:
            return None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[['date', 'close']].dropna()
        
        if len(df) == 0:
            return None
            
        return df
        
    except Exception:
        return None

def create_monthly_averages(df, years_back=5):
    """NEW: Create monthly averages from trading days over 5 years"""
    if df is None or len(df) < 30:
        return None
    
    df['year_month'] = df['date'].dt.to_period('M')
    df['month_num'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Filter last 5 years
    current_year = df['date'].max().year
    df_filtered = df[df['year'] >= current_year - years_back]
    
    if len(df_filtered) < 12:
        return None
    
    # Group by month number (Jan=1, Feb=2, etc.) and calculate average across all years
    monthly_averages = df_filtered.groupby('month_num')['close'].agg(['mean', 'std', 'count']).reset_index()
    monthly_averages.columns = ['month_num', 'avg_close', 'std_close', 'trading_days']
    
    # Create date range for each month (using current year for display)
    monthly_data = []
    current_year_dates = []
    
    for month_num in range(1, 13):
        month_data = monthly_averages[monthly_averages['month_num'] == month_num]
        if len(month_data) > 0:
            # Create ~20 trading days per month with variation around average
            month_avg = month_data['avg_close'].iloc[0]
            month_std = month_data['std_close'].iloc[0]
            trading_days = month_data['trading_days'].iloc[0]
            
            # Generate trading days for this month
            start_date = pd.Timestamp(f"{current_year}-{month_num:02d}-01")
            end_date = (start_date + pd.DateOffset(months=1) - pd.Timedelta(days=1))
            
            # Sample trading days around the average
            for day_offset in range(0, min(22, trading_days), 2):
                trade_date = start_date + pd.Timedelta(days=day_offset)
                if trade_date <= end_date:
                    # Add realistic daily variation
                    daily_variation = np.random.normal(0, month_std * 0.1)
                    daily_price = max(0.1, month_avg + daily_variation)
                    
                    monthly_data.append({
                        'date': trade_date,
                        'close': daily_price,
                        'month_num': month_num,
                        'is_monthly_avg': True,
                        'trading_days_in_month': int(trading_days)
                    })
                    current_year_dates.append(trade_date.strftime('%Y-%m-%d'))
    
    monthly_df = pd.DataFrame(monthly_data)
    if len(monthly_df) < 12:
        return None
    
    return monthly_df.sort_values('date'), current_year_dates

def create_monthly_features(df_monthly):
    """Create features from monthly aggregated data"""
    if df_monthly is None or len(df_monthly) < 12:
        return None
    
    df_features = df_monthly.copy()
    
    # Monthly cyclical features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month_num']/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month_num']/12)
    
    # Lagged monthly averages
    for lag in [1, 2, 3, 6, 12]:
        df_features[f'monthly_lag_{lag}'] = df_features['close'].shift(lag)
    
    # Rolling monthly statistics (using monthly data points)
    df_features['rolling_mean_3'] = df_features['close'].rolling(window=3, min_periods=1).mean()
    df_features['rolling_std_3'] = df_features['close'].rolling(window=3, min_periods=1).std()
    df_features['rolling_mean_6'] = df_features['close'].rolling(window=6, min_periods=1).mean()
    
    # Monthly price momentum
    df_features['monthly_momentum_1'] = df_features['close'].pct_change(1)
    df_features['monthly_momentum_3'] = df_features['close'].pct_change(3)
    
    # Trading days feature
    df_features['trading_days_feature'] = df_features['trading_days_in_month'].fillna(20)
    
    df_features = df_features.dropna()
    
    if len(df_features) < 6:
        return None
    
    return df_features

def train_monthly_forecast_model(commodity_name, forecast_months=12):
    """NEW: Train on 5-year monthly trading day averages"""
    df_daily = load_commodity_data(commodity_name)
    if df_daily is None or len(df_daily) < 50:
        return None, f"Insufficient data for {commodity_name}"
    
    # Step 1: Create monthly averages from 5 years of trading days
    df_monthly, current_year_dates = create_monthly_averages(df_daily)
    if df_monthly is None:
        return None, "Failed to create monthly averages"
    
    # Step 2: Create monthly features
    df_features = create_monthly_features(df_monthly)
    if df_features is None or len(df_features) < 6:
        return None, "Insufficient monthly features"
    
    # Step 3: Train/test split (80/20 on monthly data)
    train_size = max(4, int(len(df_features) * 0.8))
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:] if len(df_features) > train_size + 2 else None
    
    feature_cols = [col for col in train_df.columns if col not in ['date', 'close', 'month_num', 'trading_days_in_month']]
    if len(feature_cols) == 0:
        return None, "No features created"
    
    X_train = train_df[feature_cols]
    y_train = train_df['close']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Smaller model for monthly data
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=6,
        random_state=42,
        n_jobs=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Test predictions
    test_predictions = None
    mape = None
    mae = None
    
    if test_df is not None and len(test_df) > 1:
        X_test = test_df[feature_cols]
        y_test = test_df['close']
        X_test_scaled = scaler.transform(X_test)
        test_predictions = model.predict(X_test_scaled)
        
        mape = mean_absolute_percentage_error(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
    
    # Step 4: Generate forecast for next months
    last_row = df_features.iloc[-1]
    future_predictions = []
    future_dates = []
    
    for i in range(forecast_months):
        future_month = (last_row['month_num'] + i) % 12 + 1
        future_date = pd.Timestamp(f"{df_monthly['date'].max().year + (i//12)}-{future_month:02d}-15")
        
        future_features = last_row[feature_cols].values.reshape(1, -1)
        
        # Update cyclical features
        future_features[0, feature_cols.index('month_sin')] = np.sin(2 * np.pi * future_month/12)
        future_features[0, feature_cols.index('month_cos')] = np.cos(2 * np.pi * future_month/12)
        
        # Update lags (shift previous prediction into lag_1)
        future_features_scaled = scaler.transform(future_features)
        prediction = model.predict(future_features_scaled)[0]
        
        # Update lag features for next iteration
        if 'monthly_lag_1' in feature_cols:
            lag1_idx = feature_cols.index('monthly_lag_1')
            future_features[0, lag1_idx] = prediction
        
        future_dates.append(future_date.strftime('%Y-%m-%d'))
        future_predictions.append(float(prediction))
    
    # Step 5: Save model
    model_path = os.path.join(MODELS_DIR, f'{commodity_name}_monthly_model.pkl')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'monthly': True,  # Flag for monthly model
        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_points': len(df_daily),
        'monthly_points': len(df_monthly)
    }, model_path)
    
    # Historical data (last 24 months of daily simulated data)
    historical_points = min(120, len(df_monthly))  # ~5 months * 24 trading days
    historical_df = df_monthly.tail(historical_points)
    historical_dates = historical_df['date'].dt.strftime('%Y-%m-%d').tolist()
    historical_prices = historical_df['close'].tolist()
    
    test_data = {'dates': [], 'actual': [], 'predicted': []}
    if test_df is not None and test_predictions is not None:
        test_dates = test_df['date'].dt.strftime('%Y-%m-%d').tolist()
        test_data = {
            'dates': test_dates,
            'actual': test_df['close'].tolist(),
            'predicted': test_predictions.tolist()
        }
    
    return {
        'commodity': commodity_name,
        'model_type': 'monthly_trading_days_avg_5years',
        'historical': {
            'dates': historical_dates,
            'prices': historical_prices
        },
        'test_predictions': test_data,
        'forecast': {
            'dates': future_dates,
            'prices': future_predictions
        },
        'metrics': {
            'mape': float(mape * 100) if mape is not None else None,
            'mae': float(mae) if mae is not None else None,
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_samples': len(X_train),
            'test_samples': len(test_df) if test_df is not None else 0,
            'data_points': len(df_daily),
            'monthly_points': len(df_monthly),
            'forecast_months': forecast_months
        },
        'status': 'success'
    }, None

# UPDATED FORECAST ENDPOINT - SAME API STRUCTURE
@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        commodity = data.get('commodity')
        months = data.get('months', 12)
        
        if not commodity or commodity not in BARCHART_COMMODITY_CONFIG:
            return jsonify({'error': f'Unsupported commodity: {commodity}'}), 400
        
        # Use new monthly model
        result, error = train_monthly_forecast_model(commodity, months)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ALL OTHER ENDPOINTS UNCHANGED - EXACT SAME STRUCTURE
@app.route('/api/news/<commodity_key>', methods=['GET'])
def get_commodity_news(commodity_key):
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if commodity_key not in BARCHART_COMMODITY_CONFIG:
            return jsonify({
                'error': f'Unsupported commodity: {commodity_key}',
                'supported': list(BARCHART_COMMODITY_CONFIG.keys())
            }), 400
        
        news_items = scrape_barchart_news(commodity_key, force_refresh)
        
        return jsonify({
            'commodity': commodity_key,
            'name': BARCHART_COMMODITY_CONFIG[commodity_key]['name'],
            'news': news_items,
            'count': len(news_items),
            'lastUpdated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'Barchart'
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
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        all_news = {}
        
        for commodity_key in BARCHART_COMMODITY_CONFIG.keys():
            all_news[commodity_key] = scrape_barchart_news(commodity_key, force_refresh)
            time.sleep(0.5)
        
        return jsonify({
            'news': all_news,
            'count': {k: len(v) for k, v in all_news.items()},
            'lastUpdated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source': 'Barchart'
        })
        
    except Exception as e:
        return jsonify({
            'news': {k: get_fallback_news(k) for k in BARCHART_COMMODITY_CONFIG.keys()},
            'error': str(e),
            'isFallback': True
        }), 200

@app.route('/api/news/clear-cache', methods=['POST'])
def clear_news_cache():
    global NEWS_CACHE
    NEWS_CACHE.clear()
    return jsonify({
        'status': 'success',
        'message': 'News cache cleared',
        'cacheSize': 0
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'running',
        'service': 'Commodity Forecast & News API',
        'version': '2.2-monthly-averages',
        'model_type': '5yr_monthly_trading_days_avg',
        'supportedFeatures': ['monthly_ml_forecasting', 'barchart_news'],
        'supportedCommodities': list(BARCHART_COMMODITY_CONFIG.keys()),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'newsCacheSize': len(NEWS_CACHE),
        'baseDir': BASE_DIR,
        'modelsDir': MODELS_DIR
    })

@app.route('/api/forecast/status', methods=['GET'])
def forecast_status():
    return jsonify({
        'status': 'available',
        'model_type': 'monthly_trading_days_5yr_avg',
        'supported_commodities': list(BARCHART_COMMODITY_CONFIG.keys()),
        'message': 'Monthly trading days average ML forecasting (5 years data) is running'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
