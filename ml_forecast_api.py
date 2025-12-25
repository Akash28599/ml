# ml_forecast_api.py - COMPLETE RENDER-READY VERSION
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

# Barchart Commodity Configuration
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

def load_commodity_data(commodity_name):
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

def create_features(df, forecast_horizon=12):
    if df is None or len(df) < 30:
        return None
    
    df_features = df.copy()
    
    df_features['day'] = df_features['date'].dt.day
    df_features['month'] = df_features['date'].dt.month
    df_features['year'] = df_features['date'].dt.year
    df_features['day_of_week'] = df_features['date'].dt.dayofweek
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
    
    max_lag = min(20, len(df) - 1)
    for lag in [1, 2, 3, 5, 7, 10, 14, 20]:
        if lag <= max_lag:
            df_features[f'lag_{lag}'] = df_features['close'].shift(lag)
    
    window_7 = min(7, len(df) // 4)
    window_30 = min(30, len(df) // 2)
    
    if window_7 > 1:
        df_features['rolling_mean_7'] = df_features['close'].rolling(window=window_7).mean()
        df_features['rolling_std_7'] = df_features['close'].rolling(window=window_7).std()
    
    if window_30 > 1:
        df_features['rolling_mean_30'] = df_features['close'].rolling(window=window_30).mean()
        df_features['rolling_min_30'] = df_features['close'].rolling(window=window_30).min()
        df_features['rolling_max_30'] = df_features['close'].rolling(window=window_30).max()
    
    df_features['price_change_1'] = df_features['close'].pct_change(periods=1)
    if len(df) > 7:
        df_features['price_change_7'] = df_features['close'].pct_change(periods=min(7, len(df)-1))
    
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)
    
    df_features = df_features.dropna()
    
    if len(df_features) < 10:
        return None
    
    return df_features

def train_forecast_model(commodity_name, forecast_months=12):
    df = load_commodity_data(commodity_name)
    if df is None or len(df) < 50:
        return None, f"Insufficient data for {commodity_name}"
    
    df_features = create_features(df)
    if df_features is None or len(df_features) < 30:
        return None, "Failed to create features"
    
    train_size = max(30, int(len(df_features) * 0.8))
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:] if len(df_features) > train_size + 10 else None
    
    feature_cols = [col for col in train_df.columns if col not in ['date', 'close']]
    if len(feature_cols) == 0:
        return None, "No features created"
    
    X_train = train_df[feature_cols]
    y_train = train_df['close']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    n_estimators = min(50, len(X_train) // 2)
    max_depth = min(8, len(X_train) // 10)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    test_predictions = None
    mape = None
    mae = None
    
    if test_df is not None and len(test_df) > 5:
        X_test = test_df[feature_cols]
        y_test = test_df['close']
        X_test_scaled = scaler.transform(X_test)
        test_predictions = model.predict(X_test_scaled)
        
        mape = mean_absolute_percentage_error(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
    
    last_date = df['date'].max()
    future_dates = []
    future_predictions = []
    
    last_row = df_features.iloc[-1]
    current_features = last_row[feature_cols].values.reshape(1, -1)
    
    for i in range(forecast_months):
        future_date = last_date + timedelta(days=30 * (i + 1))
        future_features = current_features.copy()
        
        if 'month' in feature_cols:
            future_features[0, feature_cols.index('month')] = future_date.month
        if 'year' in feature_cols:
            future_features[0, feature_cols.index('year')] = future_date.year
        if 'month_sin' in feature_cols:
            future_features[0, feature_cols.index('month_sin')] = np.sin(2 * np.pi * future_date.month/12)
        if 'month_cos' in feature_cols:
            future_features[0, feature_cols.index('month_cos')] = np.cos(2 * np.pi * future_date.month/12)
        
        if 'day' in feature_cols:
            future_features[0, feature_cols.index('day')] = future_date.day
        if 'day_of_week' in feature_cols:
            future_features[0, feature_cols.index('day_of_week')] = future_date.weekday()
        if 'day_of_year' in feature_cols:
            future_features[0, feature_cols.index('day_of_year')] = future_date.timetuple().tm_yday
        
        future_features_scaled = scaler.transform(future_features)
        prediction = model.predict(future_features_scaled)[0]
        
        if 'lag_1' in feature_cols:
            for lag in range(20, 1, -1):
                lag_col = f'lag_{lag}'
                prev_lag_col = f'lag_{lag-1}'
                if lag_col in feature_cols and prev_lag_col in feature_cols:
                    idx_lag = feature_cols.index(lag_col)
                    idx_prev = feature_cols.index(prev_lag_col)
                    future_features[0, idx_lag] = future_features[0, idx_prev]
            
            idx_lag1 = feature_cols.index('lag_1')
            future_features[0, idx_lag1] = prediction
        
        future_dates.append(future_date.strftime('%Y-%m-%d'))
        future_predictions.append(float(prediction))
        current_features = future_features
    
    model_path = os.path.join(MODELS_DIR, f'{commodity_name}_model.pkl')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_date': last_date.strftime('%Y-%m-%d')
    }, model_path)
    
    historical_points = min(100, len(df))
    historical_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()[-historical_points:]
    historical_prices = df['close'].tolist()[-historical_points:]
    
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
            'data_points': len(df),
            'forecast_months': forecast_months
        },
        'status': 'success'
    }, None

# API ENDPOINTS
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
        
        result, error = train_forecast_model(commodity, months)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        'version': '2.1-render',
        'supportedFeatures': ['ml_forecasting', 'barchart_news'],
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
        'supported_commodities': list(BARCHART_COMMODITY_CONFIG.keys()),
        'message': 'Machine learning forecasting API is running'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
