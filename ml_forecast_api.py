# ml_forecast_api.py - MONTHLY AVERAGE FORECASTING
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
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
MONTHLY_MODELS_DIR = os.path.join(BASE_DIR, 'monthly_models')
os.makedirs(MONTHLY_MODELS_DIR, exist_ok=True)

BARCHART_COMMODITY_CONFIG = {
    'wheat': {'name': 'Wheat CBOT', 'csv_file': 'wheat.csv', 'icon': '🌾'},
    'milling_wheat': {'name': 'Milling Wheat', 'csv_file': 'millingwheat.csv', 'icon': '🌾'},
    'palm': {'name': 'Palm Oil', 'csv_file': 'palmoil.csv', 'icon': '🌴'},
    'sugar': {'name': 'Sugar', 'csv_file': 'sugar.csv', 'icon': '🍬'},
    'aluminum': {'name': 'Aluminum', 'csv_file': 'alumnium.csv', 'icon': '🥫'},
    'crude_palm': {'name': 'Brent Crude Oil', 'csv_file': 'brentcrude.csv', 'icon': '🛢️'}
}

# ==================== MONTHLY AVERAGE DATA PROCESSING ====================
def calculate_monthly_averages(commodity_name):
    """
    1. Load daily data
    2. For each month: take all trading days, calculate average
    3. Return monthly averages for training
    """
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
    
    # Find CSV file
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
        # Load daily data
        df = pd.read_csv(csv_path, header=None)
        
        # Parse columns
        if df.shape[1] >= 7:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        elif df.shape[1] >= 6:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close']
        else:
            return None
        
        # Parse dates and prices
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df[['date', 'close']].dropna().sort_values('date')
        
        if len(df) < 100:  # Need at least ~100 trading days
            return None
        
        # ==================== KEY LOGIC ====================
        # Calculate monthly averages from daily data
        df.set_index('date', inplace=True)
        
        # Group by month and calculate average of all trading days
        monthly_avg = df['close'].resample('M').agg(['mean', 'count']).reset_index()
        monthly_avg.columns = ['month_year', 'monthly_avg_price', 'trading_days']
        
        # Filter months with at least 15 trading days
        monthly_avg = monthly_avg[monthly_avg['trading_days'] >= 15]
        
        # Get last 5 years of data (60 months)
        monthly_avg = monthly_avg.tail(60)
        
        if len(monthly_avg) < 12:  # Need at least 1 year
            return None
        
        # Extract month and year
        monthly_avg['year'] = monthly_avg['month_year'].dt.year
        monthly_avg['month'] = monthly_avg['month_year'].dt.month
        
        # Calculate month name for display
        monthly_avg['month_name'] = monthly_avg['month_year'].dt.strftime('%b %Y')
        
        return monthly_avg[['month_year', 'monthly_avg_price', 'trading_days', 'year', 'month', 'month_name']]
        
    except Exception as e:
        print(f"Error calculating monthly averages for {commodity_name}: {str(e)}")
        return None

def create_monthly_features(monthly_data):
    """
    Create features for monthly time series forecasting
    Each row = one month's average price
    """
    df = monthly_data.copy()
    
    # Basic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (previous months' averages)
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['monthly_avg_price'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'rolling_mean_{window}'] = df['monthly_avg_price'].rolling(window=window).mean().shift(1)
        df[f'rolling_std_{window}'] = df['monthly_avg_price'].rolling(window=window).std().shift(1)
    
    # Year-over-year change
    df['yoy_change'] = df['monthly_avg_price'].pct_change(periods=12)
    
    # Month-over-month change
    df['mom_change'] = df['monthly_avg_price'].pct_change(periods=1)
    
    # Quarter feature
    df['quarter'] = (df['month'] - 1) // 3 + 1
    
    # Seasonality (for agricultural commodities)
    seasons = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['season'] = df['month'].map(seasons)
    
    # Drop NaN rows
    df = df.dropna()
    
    if len(df) < 12:
        return None
    
    return df

def train_monthly_average_model(commodity_name, forecast_months=12):
    """
    Train model on monthly average prices
    Predict future monthly averages
    """
    print(f"Training monthly average model for {commodity_name}...")
    
    # 1. Calculate monthly averages from daily data
    monthly_data = calculate_monthly_averages(commodity_name)
    if monthly_data is None:
        return None, f"Cannot calculate monthly averages for {commodity_name}"
    
    print(f"Monthly data: {len(monthly_data)} months")
    print(f"Date range: {monthly_data['month_year'].min()} to {monthly_data['month_year'].max()}")
    
    # 2. Create features
    df_features = create_monthly_features(monthly_data)
    if df_features is None:
        return None, "Failed to create monthly features"
    
    # 3. Prepare training data
    feature_cols = [col for col in df_features.columns 
                   if col not in ['month_year', 'monthly_avg_price', 'trading_days', 
                                 'year', 'month', 'month_name', 'season']]
    
    if not feature_cols:
        return None, "No features created"
    
    X = df_features[feature_cols]
    y = df_features['monthly_avg_price']
    
    # Train/test split (80/20)
    train_size = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:] if len(df_features) > train_size + 3 else None
    
    X_train = train_df[feature_cols]
    y_train = train_df['monthly_avg_price']
    
    # 4. Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    test_predictions = None
    mape = None
    mae = None
    
    if test_df is not None and len(test_df) >= 3:
        X_test = test_df[feature_cols]
        y_test = test_df['monthly_avg_price']
        X_test_scaled = scaler.transform(X_test)
        test_predictions = model.predict(X_test_scaled)
        
        mape = mean_absolute_percentage_error(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
    
    # 6. Predict future months
    last_date = df_features['month_year'].max()
    future_months = []
    future_predictions = []
    
    # Start with last known data
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
        
        # Create date (last day of month)
        last_day = calendar.monthrange(next_year, next_month)[1]
        future_date = datetime(next_year, next_month, last_day)
        
        # Update features for future month
        future_features = current_features.copy()
        
        # Update month and year in features
        for col, idx in [(col, feature_cols.index(col)) for col in feature_cols if col in ['month', 'year', 'quarter', 'month_sin', 'month_cos']]:
            if col == 'month':
                future_features[0, idx] = next_month
            elif col == 'year':
                future_features[0, idx] = next_year
            elif col == 'quarter':
                future_features[0, idx] = (next_month - 1) // 3 + 1
            elif col == 'month_sin':
                future_features[0, idx] = np.sin(2 * np.pi * next_month / 12)
            elif col == 'month_cos':
                future_features[0, idx] = np.cos(2 * np.pi * next_month / 12)
        
        # Make prediction
        future_features_scaled = scaler.transform(future_features)
        prediction = model.predict(future_features_scaled)[0]
        
        # Update lag features for next iteration
        for lag in range(12, 1, -1):
            lag_col = f'lag_{lag}'
            prev_lag_col = f'lag_{lag-1}'
            if lag_col in feature_cols and prev_lag_col in feature_cols:
                idx_lag = feature_cols.index(lag_col)
                idx_prev = feature_cols.index(prev_lag_col)
                future_features[0, idx_lag] = future_features[0, idx_prev]
        
        # Update first lag with current prediction
        if 'lag_1' in feature_cols:
            idx = feature_cols.index('lag_1')
            future_features[0, idx] = prediction
        
        future_months.append(future_date.strftime('%Y-%m-%d'))
        future_predictions.append(float(prediction))
        current_features = future_features
        last_date = future_date
    
    # 7. Prepare response
    historical_data = []
    for _, row in monthly_data.iterrows():
        historical_data.append({
            'month': row['month_year'].strftime('%Y-%m-%d'),
            'month_name': row['month_name'],
            'monthly_avg': float(row['monthly_avg_price']),
            'trading_days': int(row['trading_days'])
        })
    
    return {
        'commodity': commodity_name,
        'name': BARCHART_COMMODITY_CONFIG[commodity_name]['name'],
        'forecast_type': 'monthly_average',
        'method': 'Trained on monthly averages (calculated from daily trading days)',
        
        'historical': {
            'months': historical_data,
            'total_months': len(monthly_data),
            'date_range': {
                'start': monthly_data['month_year'].min().strftime('%Y-%m-%d'),
                'end': monthly_data['month_year'].max().strftime('%Y-%m-%d')
            }
        },
        
        'forecast': {
            'months': future_months,
            'month_names': [datetime.strptime(m, '%Y-%m-%d').strftime('%b %Y') for m in future_months],
            'predicted_averages': future_predictions
        },
        
        'metrics': {
            'mape_percent': float(mape * 100) if mape else None,
            'mae': float(mae) if mae else None,
            'training_months': len(X_train),
            'test_months': len(test_df) if test_df else 0,
            'data_months': len(monthly_data),
            'avg_trading_days_per_month': monthly_data['trading_days'].mean(),
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        
        'explanation': {
            'training_method': 'Model trained on historical monthly average prices',
            'calculation': 'Each monthly average = (sum of daily closing prices) / (number of trading days in that month)',
            'prediction': 'Forecasts future monthly average prices',
            'comparison': 'Each forecasted value is comparable to historical monthly averages'
        },
        
        'status': 'success'
    }, None

# ==================== API ENDPOINTS ====================
@app.route('/')
def home():
    return jsonify({
        'service': 'Monthly Average Commodity Forecasting API',
        'version': '1.0',
        'description': 'Trains on monthly averages calculated from daily trading data',
        'endpoints': {
            '/api/forecast/monthly-average': 'POST - Forecast monthly average prices',
            '/api/test-data/<commodity>': 'GET - View monthly average data'
        }
    })

@app.route('/api/forecast/monthly-average', methods=['POST'])
def forecast_monthly_average():
    """Forecast monthly average prices"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        commodity = data.get('commodity')
        months = data.get('months', 12)
        
        if not commodity:
            return jsonify({
                'error': 'Missing commodity parameter',
                'supported': list(BARCHART_COMMODITY_CONFIG.keys())
            }), 400
        
        if commodity not in BARCHART_COMMODITY_CONFIG:
            return jsonify({
                'error': f'Unsupported commodity: {commodity}',
                'supported': list(BARCHART_COMMODITY_CONFIG.keys())
            }), 400
        
        result, error = train_monthly_average_model(commodity, months)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-data/<commodity_name>', methods=['GET'])
def test_monthly_data(commodity_name):
    """Test endpoint to see monthly average calculations"""
    if commodity_name not in BARCHART_COMMODITY_CONFIG:
        return jsonify({'error': 'Unsupported commodity'}), 400
    
    monthly_data = calculate_monthly_averages(commodity_name)
    
    if monthly_data is None:
        return jsonify({
            'commodity': commodity_name,
            'status': 'error',
            'message': 'Could not calculate monthly averages'
        }), 400
    
    # Convert to list for JSON response
    data_list = []
    for _, row in monthly_data.iterrows():
        data_list.append({
            'month': row['month_year'].strftime('%Y-%m'),
            'month_name': row['month_name'],
            'monthly_avg_price': float(row['monthly_avg_price']),
            'trading_days': int(row['trading_days']),
            'year': int(row['year']),
            'month_num': int(row['month'])
        })
    
    return jsonify({
        'commodity': commodity_name,
        'total_months': len(monthly_data),
        'date_range': {
            'start': monthly_data['month_year'].min().strftime('%Y-%m'),
            'end': monthly_data['month_year'].max().strftime('%Y-%m')
        },
        'monthly_data': data_list,
        'stats': {
            'avg_price': float(monthly_data['monthly_avg_price'].mean()),
            'min_price': float(monthly_data['monthly_avg_price'].min()),
            'max_price': float(monthly_data['monthly_avg_price'].max()),
            'avg_trading_days': float(monthly_data['trading_days'].mean())
        }
    })

@app.route('/api/commodities', methods=['GET'])
def list_commodities():
    """List all supported commodities"""
    commodities = []
    for key, config in BARCHART_COMMODITY_CONFIG.items():
        commodities.append({
            'id': key,
            'name': config['name'],
            'icon': config.get('icon', '📊'),
            'csv_file': config['csv_file']
        })
    
    return jsonify({
        'commodities': commodities,
        'count': len(commodities)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("=" * 50)
    print("MONTHLY AVERAGE COMMODITY FORECASTING API")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Commodities: {list(BARCHART_COMMODITY_CONFIG.keys())}")
    print(f"Method: Trains on monthly averages calculated from daily trading days")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=True)