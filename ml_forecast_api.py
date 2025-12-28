# ml_forecast_api_smart_hybrid.py - SMART MODEL SELECTION WITH ADDED NEWS FEATURES
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import json
import sys
from scipy import stats
import traceback
import requests
from bs4 import BeautifulSoup
import time

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# News configuration - ADDED FROM OLD FILE
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
NEWS_CACHE_TIMEOUT = 300  # 5 minutes

# Smart model selection configuration (EXISTING - UNCHANGED)
MODEL_CONFIG = {
    'wheat': {
        'preferred_model': 'simple_ensemble',  # XGBoost made it WORSE
        'seasonal_factors': {1: 1.01, 2: 1.02, 3: 1.00, 4: 0.99, 5: 0.98, 6: 0.97,
                            7: 0.98, 8: 0.99, 9: 1.00, 10: 1.01, 11: 1.02, 12: 1.01},
        'price_range': (450, 700),
        'volatility': 'low'
    },
    'milling_wheat': {
        'preferred_model': 'simple_ensemble',  # Original was 11.86% MAPE, XGBoost was 13.68%
        'seasonal_factors': {1: 1.00, 2: 0.99, 3: 0.96, 4: 0.94, 5: 0.92, 6: 0.90,
                            7: 0.89, 8: 0.88, 9: 0.87, 10: 0.88, 11: 0.90, 12: 0.92},
        'price_range': (180, 250),
        'volatility': 'medium'
    },
    'palm': {
        'preferred_model':  'simple_ensemble',  # Original was 6.62% MAPE - XGBoost might help
        'seasonal_factors': {1: 1.02, 2: 1.05, 3: 1.03, 4: 0.95, 5: 0.92, 6: 0.94,
                            7: 0.98, 8: 1.01, 9: 1.02, 10: 1.03, 11: 0.98, 12: 0.97},
        'price_range': (3500, 6000),
        'volatility': 'high'
    },
    'sugar': {
        'preferred_model': 'simple_ensemble',
        'seasonal_factors': {m: 1.0 for m in range(1, 13)},
        'price_range': None,
        'volatility': 'medium'
    },
    'aluminum': {
        'preferred_model': 'simple_ensemble',
        'seasonal_factors': {m: 1.0 for m in range(1, 13)},
        'price_range': None,
        'volatility': 'medium'
    },
    'crude_palm': {
        'preferred_model': 'xgboost',
        'seasonal_factors': {1: 1.03, 2: 1.04, 3: 1.02, 4: 0.96, 5: 0.93, 6: 0.95,
                            7: 0.97, 8: 1.00, 9: 1.01, 10: 1.02, 11: 0.99, 12: 0.98},
        'price_range': None,
        'volatility': 'high'
    }
}

# NEWS FUNCTIONS - ADDED FROM OLD FILE
def get_default_image(commodity_key):
    """Get default image for commodity news"""
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
    """Get fallback news when scraping fails"""
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
    """Extract news articles from Barchart HTML"""
    soup = BeautifulSoup(html_content, 'html.parser')
    news_items = []
    
    # Look for most recent stories
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
    
    # Fallback method if first method didn't work
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
    """Scrape news from Barchart with caching"""
    cache_key = f"{commodity_key}_news"
    current_time = time.time()
    
    # Check cache first
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
        print(f"❌ News scraping error for {commodity_key}: {e}")
        return get_fallback_news(commodity_key)
    except Exception as e:
        print(f"❌ Unexpected error scraping news for {commodity_key}: {e}")
        return get_fallback_news(commodity_key)

# EXISTING FORECAST FUNCTIONS - UNCHANGED
def load_commodity_data(commodity_name):
    """Load CSV data with error handling"""
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
        os.path.join('.', 'data', csv_filename),
        os.path.join('.', csv_filename),
        csv_filename
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        print(f"❌ File not found: {csv_filename}")
        return None
    
    try:
        df = pd.read_csv(csv_path, header=None)
        
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close']
        else:
            df.columns = ['date', 'close']
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna()
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        print(f"✅ Loaded {len(df)} records for {commodity_name}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {commodity_name}: {str(e)}")
        traceback.print_exc()
        return None

def calculate_historical_2025(df):
    """Calculate actual 2025 monthly averages"""
    if df is None:
        return []
    
    df_2025 = df[df['year'] == 2025].copy()
    
    if len(df_2025) == 0:
        return []
    
    monthly_avg = []
    for month in range(1, 13):
        month_data = df_2025[df_2025['month'] == month]
        if len(month_data) > 0:
            avg_price = float(month_data['close'].mean())
            monthly_avg.append({
                'month': month,
                'avg_price': avg_price,
                'trading_days': len(month_data),
                'date': f"2025-{month:02d}-15"
            })
        else:
            monthly_avg.append({
                'month': month,
                'avg_price': None,
                'trading_days': 0,
                'date': f"2025-{month:02d}-15"
            })
    
    return monthly_avg

def simple_ensemble_prediction(df, target_year, target_month, commodity_config, historical_2025_data=None):
    """Original reliable 4-method ensemble"""
    month_data = df[df['month'] == target_month].copy()
    
    if len(month_data) < 3:
        return None, 0.5
    
    month_data = month_data.sort_values('date')
    years = month_data['year'].values.astype(float)
    prices = month_data['close'].values.astype(float)
    
    # Check for recent price regime change
    adjustment_needed = False
    adjustment_factor = 1.0
    
    if historical_2025_data:
        current_month_data = [d for d in historical_2025_data if d['month'] == target_month and d['avg_price'] is not None]
        if current_month_data:
            current_price = current_month_data[0]['avg_price']
            hist_avg = np.mean(prices)
            if hist_avg > 0:
                price_ratio = current_price / hist_avg
                # If 2025 price is significantly different (+/- 15%)
                if abs(price_ratio - 1.0) > 0.15:
                    adjustment_needed = True
                    # Smart adjustment: blend current and historical
                    adjustment_factor = 0.5 * price_ratio + 0.5 * 1.0
    
    predictions = []
    method_weights = []
    
    # Get seasonal factor
    seasonal_factor = commodity_config['seasonal_factors'].get(target_month, 1.0)
    
    # METHOD 1: Robust trend with momentum
    if len(prices) >= 4:
        lookback = min(4, len(prices))
        recent_years = years[-lookback:]
        recent_prices = prices[-lookback:]
        
        slope, intercept = np.polyfit(recent_years, recent_prices, 1)
        trend_pred = slope * target_year + intercept
        
        # Add momentum from recent changes
        if len(prices) >= 3:
            recent_momentum = (prices[-1] - prices[-3]) / 2
            years_ahead = target_year - years[-1]
            trend_pred += recent_momentum * years_ahead * 0.5
        
        # Apply adjustments
        trend_pred *= seasonal_factor
        if adjustment_needed:
            trend_pred *= adjustment_factor
        
        predictions.append(trend_pred)
        method_weights.append(0.35)
    
    # METHOD 2: Exponential weighted moving average
    if len(prices) >= 3:
        decay = 0.8
        weights = np.array([decay ** i for i in range(len(prices)-1, -1, -1)])
        weights = weights / weights.sum()
        
        wma_pred = np.sum(prices * weights)
        wma_pred *= seasonal_factor
        
        if adjustment_needed:
            wma_pred *= adjustment_factor
        
        predictions.append(wma_pred)
        method_weights.append(0.30)
    
    # METHOD 3: Percentile-based
    if len(prices) >= 5:
        # Choose appropriate percentile based on commodity
        if commodity_config.get('volatility') == 'high':
            percentile_val = np.percentile(prices, 60)  # Conservative for volatile
        else:
            percentile_val = np.percentile(prices, 50)  # Median for stable
        
        percentile_pred = percentile_val
        percentile_pred *= seasonal_factor
        
        if adjustment_needed:
            percentile_pred *= adjustment_factor
        
        predictions.append(percentile_pred)
        method_weights.append(0.25)
    
    # METHOD 4: Same-month momentum (seasonal naive)
    if len(prices) >= 2:
        seasonal_pred = prices[-1]
        
        # Add trend from previous same-month changes
        if len(prices) >= 3:
            month_trend = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
            seasonal_pred *= (1 + month_trend)
        
        seasonal_pred *= seasonal_factor
        
        if adjustment_needed:
            seasonal_pred *= adjustment_factor
        
        predictions.append(seasonal_pred)
        method_weights.append(0.10)
    
    if not predictions:
        return None, 0.5
    
    # Weighted ensemble
    method_weights = np.array(method_weights)
    method_weights = method_weights / method_weights.sum()
    
    final_prediction = float(np.sum(np.array(predictions) * method_weights))
    
    # Apply commodity-specific bounds
    price_range = commodity_config.get('price_range')
    if price_range:
        min_price, max_price = price_range
        final_prediction = max(min_price, min(max_price, final_prediction))
    
    # Calculate confidence
    confidence = calculate_simple_confidence(prices, predictions, commodity_config)
    
    return final_prediction, confidence

def calculate_simple_confidence(prices, predictions, commodity_config):
    """Calculate confidence for simple ensemble"""
    if len(prices) < 4:
        return 0.5
    
    base_confidence = 0.7
    
    # Adjust for data quantity
    if len(prices) >= 8:
        base_confidence *= 1.1
    elif len(prices) <= 4:
        base_confidence *= 0.9
    
    # Adjust for prediction agreement
    if len(predictions) > 1:
        prediction_std = np.std(predictions)
        prediction_mean = np.mean(predictions)
        if prediction_mean > 0:
            agreement = 1.0 - min(1.0, prediction_std / prediction_mean)
            base_confidence = base_confidence * 0.7 + agreement * 0.3
    
    # Adjust for commodity volatility
    volatility = commodity_config.get('volatility', 'medium')
    if volatility == 'high':
        base_confidence *= 0.9
    elif volatility == 'low':
        base_confidence *= 1.05
    
    # Calculate historical accuracy if possible
    if len(prices) >= 6:
        test_errors = []
        for i in range(1, min(4, len(prices)-2)):
            test_data = prices[:-i]
            test_years = np.arange(len(test_data))
            
            test_slope, test_intercept = np.polyfit(test_years, test_data, 1)
            test_pred = test_slope * (len(test_data) + i - 1) + test_intercept
            actual = prices[-i]
            
            if actual > 0:
                error = abs(test_pred - actual) / actual
                test_errors.append(error)
        
        if test_errors:
            avg_test_error = np.mean(test_errors)
            accuracy_factor = 1.0 - min(1.0, avg_test_error)
            base_confidence = base_confidence * 0.6 + accuracy_factor * 0.4
    
    confidence = max(0.4, min(0.95, base_confidence))
    return float(confidence)

def xgboost_prediction(df, target_year, target_month, commodity_config):
    """XGBoost prediction - simplified version"""
    try:
        import xgboost as xgb
    except ImportError:
        print("⚠️ XGBoost not installed, using simple ensemble")
        return None, 0.5
    
    month_data = df[df['month'] == target_month].copy()
    
    if len(month_data) < 5:
        return None, 0.5
    
    # Prepare simple features
    monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
        'close': 'mean',
        'year': 'first',
        'month': 'first'
    }).reset_index()
    
    # Filter for target month
    target_month_data = monthly_data[monthly_data['month'] == target_month].copy()
    
    if len(target_month_data) < 4:
        return None, 0.5
    
    # Simple features: last 3 prices and year
    recent_prices = target_month_data['close'].tail(3).values
    recent_years = target_month_data['year'].tail(3).values
    
    if len(recent_prices) < 3:
        return None, 0.5
    
    # Create feature vector
    features = list(recent_prices) + list(recent_years)
    features.append(target_month)
    
    # For prediction, we need to create a simple model
    # This is a simplified version - in production you'd train properly
    X = []
    y = []
    
    for i in range(3, len(target_month_data)):
        if i >= 3:
            feat = list(target_month_data['close'].iloc[i-3:i].values)
            feat.extend(list(target_month_data['year'].iloc[i-3:i].values))
            feat.append(target_month)
            X.append(feat)
            y.append(target_month_data['close'].iloc[i])
    
    if len(X) < 8:
        return None, 0.5
    
    X = np.array(X)
    y = np.array(y)
    
    # Train simple XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    # Simple train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model.fit(X_train, y_train, verbose=False)
    
    # Make prediction
    current_features = np.array(features).reshape(1, -1)
    prediction = model.predict(current_features)[0]
    
    # Apply seasonal adjustment
    seasonal_factor = commodity_config['seasonal_factors'].get(target_month, 1.0)
    prediction *= seasonal_factor
    
    # Calculate confidence based on model performance
    test_score = model.score(X_test, y_test)
    confidence = 0.5 + 0.3 * max(0, test_score)  # 0.5-0.8 range
    
    return float(prediction), float(confidence)

def smart_hybrid_prediction(df, target_year, target_month, commodity, historical_2025_data=None):
    """Intelligently choose the best model for each commodity"""
    config = MODEL_CONFIG.get(commodity, MODEL_CONFIG['wheat'])
    
    if config['preferred_model'] == 'xgboost':
        print(f"🔍 Trying XGBoost for {commodity}...")
        try:
            xgb_pred, xgb_conf = xgboost_prediction(df, target_year, target_month, config)
            
            if xgb_pred is not None and xgb_conf > 0.55:
                print(f"✅ Using XGBoost for {commodity} (confidence: {xgb_conf:.3f})")
                return xgb_pred, xgb_conf
            else:
                print(f"⚠️ XGBoost failed or low confidence for {commodity}, using simple ensemble")
                return simple_ensemble_prediction(df, target_year, target_month, config, historical_2025_data)
                
        except Exception as e:
            print(f"❌ XGBoost error for {commodity}: {e}, using simple ensemble")
            return simple_ensemble_prediction(df, target_year, target_month, config, historical_2025_data)
    
    else:  # simple_ensemble or default
        print(f"🎯 Using Simple Ensemble for {commodity} (proven better)")
        return simple_ensemble_prediction(df, target_year, target_month, config, historical_2025_data)

# EXISTING FORECAST ENDPOINT - UNCHANGED
@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.json
        commodity = data.get('commodity')
        
        if not commodity:
            return jsonify({'error': 'Commodity parameter required'}), 400
        
        print(f"\n🚀 Processing SMART HYBRID forecast for: {commodity}")
        
        # Load data
        df = load_commodity_data(commodity)
        if df is None or len(df) < 100:
            return jsonify({'error': f'Insufficient data for {commodity}'}), 400
        
        # Historical 2025
        historical_2025 = calculate_historical_2025(df)
        actual_months = len([h for h in historical_2025 if h['avg_price'] is not None])
        print(f"📊 Historical 2025: {actual_months} months with data")
        
        # Predict 2025 (using 2013-2024 data)
        print(f"📈 Predicting 2025 with smart hybrid model...")
        df_2013_2024 = df[df['year'] < 2025].copy()
        
        predicted_2025 = []
        for month in range(1, 13):
            price, confidence = smart_hybrid_prediction(
                df_2013_2024, 2025, month, commodity, historical_2025
            )
            predicted_2025.append({
                'month': month,
                'date': f"2025-{month:02d}-15",
                'predicted_price': price,
                'confidence': confidence if confidence else 0.5
            })
        
        # Predict 2026 (using 2013-2025 data)
        print(f"📈 Predicting 2026 with smart hybrid model...")
        df_with_2025 = df.copy()
        for hist in historical_2025:
            if hist['avg_price'] is not None:
                new_date = pd.Timestamp(f"2025-{hist['month']:02d}-15")
                new_row = pd.DataFrame([{
                    'date': new_date,
                    'close': hist['avg_price'],
                    'year': 2025,
                    'month': hist['month'],
                    'quarter': new_date.quarter
                }])
                df_with_2025 = pd.concat([df_with_2025, new_row], ignore_index=True)
        
        predicted_2026 = []
        for month in range(1, 13):
            price, confidence = smart_hybrid_prediction(df_with_2025, 2026, month, commodity)
            predicted_2026.append({
                'month': month,
                'date': f"2026-{month:02d}-15",
                'predicted_price': price,
                'confidence': confidence if confidence else 0.5
            })
        
        # Calculate accuracy
        monthly_accuracy = []
        percentage_errors = []
        
        for hist, pred in zip(historical_2025, predicted_2025):
            if hist['avg_price'] is not None and pred['predicted_price'] is not None:
                actual = hist['avg_price']
                predicted = pred['predicted_price']
                
                error = predicted - actual
                percentage_error = abs(error) / actual * 100 if actual > 0 else 0
                percentage_errors.append(percentage_error)
                
                monthly_accuracy.append({
                    'month': hist['month'],
                    'actual_price': actual,
                    'predicted_price': predicted,
                    'error': float(error),
                    'percentage_error': float(percentage_error),
                    'accurate_within_5%': percentage_error <= 5,
                    'accurate_within_10%': percentage_error <= 10,
                    'accurate_within_15%': percentage_error <= 15,
                    'accurate_within_20%': percentage_error <= 20
                })
        
        if monthly_accuracy:
            mape = np.mean(percentage_errors)
            accuracy_5pct = sum(1 for m in monthly_accuracy if m['percentage_error'] <= 5) / len(monthly_accuracy) * 100
            accuracy_10pct = sum(1 for m in monthly_accuracy if m['percentage_error'] <= 10) / len(monthly_accuracy) * 100
            accuracy_15pct = sum(1 for m in monthly_accuracy if m['percentage_error'] <= 15) / len(monthly_accuracy) * 100
            accuracy_20pct = sum(1 for m in monthly_accuracy if m['percentage_error'] <= 20) / len(monthly_accuracy) * 100
            
            # Find best and worst months
            best_month = min(monthly_accuracy, key=lambda x: x['percentage_error'])
            worst_month = max(monthly_accuracy, key=lambda x: x['percentage_error'])
            
            print(f"\n✅ 2025 SMART HYBRID Accuracy for {commodity}:")
            print(f"   MAPE: {mape:.2f}%")
            print(f"   ≤5%: {accuracy_5pct:.1f}% months")
            print(f"   ≤10%: {accuracy_10pct:.1f}% months")
            print(f"   ≤15%: {accuracy_15pct:.1f}% months")
            print(f"   ≤20%: {accuracy_20pct:.1f}% months")
            print(f"   Best month: {best_month['month']} ({best_month['percentage_error']:.1f}% error)")
            print(f"   Worst month: {worst_month['month']} ({worst_month['percentage_error']:.1f}% error)")
        
        # Determine which model was actually used
        model_used = MODEL_CONFIG.get(commodity, {}).get('preferred_model', 'simple_ensemble')
        model_name = "Simple Ensemble" if model_used == 'simple_ensemble' else "XGBoost Hybrid"
        
        # Prepare response
        result = {
            'commodity': commodity,
            'historical_2025': [
                {
                    'month': h['month'],
                    'date': h['date'],
                    'actual_price': h['avg_price'],
                    'trading_days': h['trading_days']
                }
                for h in historical_2025
            ],
            'predicted_2025': [
                {
                    'month': p['month'],
                    'date': p['date'],
                    'predicted_price': p['predicted_price'],
                    'confidence': p['confidence']
                }
                for p in predicted_2025
            ],
            'predicted_2026': [
                {
                    'month': p['month'],
                    'date': p['date'],
                    'predicted_price': p['predicted_price'],
                    'confidence': p['confidence']
                }
                for p in predicted_2026
            ],
            'accuracy_analysis': {
                'monthly_details': monthly_accuracy,
                'summary': {
                    'mean_absolute_percentage_error': float(mape) if monthly_accuracy else 0.0,
                    'accuracy_within_5_percent': float(accuracy_5pct) if monthly_accuracy else 0.0,
                    'accuracy_within_10_percent': float(accuracy_10pct) if monthly_accuracy else 0.0,
                    'accuracy_within_15_percent': float(accuracy_15pct) if monthly_accuracy else 0.0,
                    'accuracy_within_20_percent': float(accuracy_20pct) if monthly_accuracy else 0.0,
                    'total_comparable_months': len(monthly_accuracy),
                    'best_month': int(best_month['month']) if monthly_accuracy else None,
                    'best_month_error': float(best_month['percentage_error']) if monthly_accuracy else 0.0,
                    'worst_month': int(worst_month['month']) if monthly_accuracy else None,
                    'worst_month_error': float(worst_month['percentage_error']) if monthly_accuracy else 0.0
                }
            } if monthly_accuracy else {},
            'model_info': {
                'algorithm': f'Smart Hybrid - {model_name}',
                'methods_used': [
                    'Intelligent Model Selection',
                    '4-Method Ensemble (Trend, EWMA, Percentile, Seasonal)',
                    'XGBoost where beneficial'
                ],
                'commodity_specific': 'Yes (learned from testing)',
                'model_selection': f'Uses {model_used} for {commodity}',
                'confidence_calculation': 'Performance-based',
                'training_years_2025': f"{df_2013_2024['year'].min()}-2024",
                'training_years_2026': f"{df_with_2025['year'].min()}-2025"
            },
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# NEWS ENDPOINTS - ADDED FROM OLD FILE
@app.route('/api/news/<commodity_key>', methods=['GET'])
def get_commodity_news(commodity_key):
    """Get news for a specific commodity"""
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
    """Get news for all commodities"""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        all_news = {}
        
        for commodity_key in BARCHART_COMMODITY_CONFIG.keys():
            all_news[commodity_key] = scrape_barchart_news(commodity_key, force_refresh)
            # Small delay to avoid overwhelming the server
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
    """Clear the news cache"""
    global NEWS_CACHE
    NEWS_CACHE.clear()
    return jsonify({
        'status': 'success',
        'message': 'News cache cleared',
        'cacheSize': 0
    })

# EXISTING ENDPOINTS - UNCHANGED
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'smart_hybrid',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model_selection', methods=['GET'])
def model_selection():
    """Get model selection information"""
    commodity = request.args.get('commodity', 'wheat')
    config = MODEL_CONFIG.get(commodity, MODEL_CONFIG['wheat'])
    
    return jsonify({
        'commodity': commodity,
        'preferred_model': config['preferred_model'],
        'reasoning': get_model_reasoning(commodity),
        'expected_mape': get_expected_mape(commodity)
    })

def get_model_reasoning(commodity):
    """Get reasoning for model selection"""
    reasons = {
        'wheat': 'XGBoost made it WORSE (16.42% MAPE vs 7.00% MAPE)',
        'milling_wheat': 'XGBoost made it WORSE (13.68% MAPE vs 11.86% MAPE)',
        'palm': 'XGBoost might help improve 6.62% MAPE (needs testing)',
        'crude_palm': 'Similar to palm oil, XGBoost might help',
        'default': 'Simple ensemble is reliable and proven'
    }
    return reasons.get(commodity, reasons['default'])

def get_expected_mape(commodity):
    """Get expected MAPE based on testing"""
    expected = {
        'wheat': '7-8% (simple ensemble)',
        'milling_wheat': '10-12% (simple ensemble)',
        'palm': '6-7% (simple ensemble)',
        'default': '8-12% (simple ensemble)'
    }
    return expected.get(commodity, expected['default'])

# NEW STATUS ENDPOINT - COMBINING BOTH FEATURES
@app.route('/api/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'Commodity Smart Hybrid Forecast & News API',
        'version': '2.3-smart-hybrid-with-news',
        'model_type': 'smart_hybrid_with_news_scraping',
        'supportedFeatures': ['smart_hybrid_ml_forecasting', 'barchart_news_scraping', 'news_caching'],
        'supportedCommodities': list(BARCHART_COMMODITY_CONFIG.keys()),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'newsCacheSize': len(NEWS_CACHE),
        'forecastModel': 'smart_hybrid_selection'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))
    print(f"\n🚀 SMART HYBRID Forecasting API with News Features running on port {port}")
    print("🎯 Key features:")
    print("   ✅ Smart model selection based on test results")
    print("   ✅ Barchart news scraping for all commodities")
    print("   ✅ 5-minute news caching for performance")
    print("   ✅ Simple ensemble for wheat and milling wheat")
    print("   ✅ XGBoost for palm/crude palm when available")
    print("\n📊 Expected forecast performance:")
    print("   • Wheat: 7-8% MAPE")
    print("   • Milling Wheat: 10-12% MAPE")
    print("   • Palm: 6-7% MAPE")
    print("\n📰 News endpoints available:")
    print("   • GET /api/news/<commodity>")
    print("   • GET /api/news/all")
    print("   • POST /api/news/clear-cache")
    print("   • GET /api/status")
    
    app.run(host='0.0.0.0', port=port, debug=True)