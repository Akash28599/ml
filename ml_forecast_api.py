# ml_forecast_api.py - WITH JOBLIB FIX & RENDER COMPATIBILITY
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Monkey patch for joblib compatibility with Python 3.14
try:
    import ast
    # Fix for Python 3.14 AST changes
    if not hasattr(ast, 'Num'):
        ast.Num = type('Num', (), {})
    if not hasattr(ast, 'Str'):
        ast.Str = type('Str', (), {})
    if not hasattr(ast, 'NameConstant'):
        ast.NameConstant = type('NameConstant', (), {})
except:
    pass

app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base directory: {BASE_DIR}")

# Create models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models directory: {MODELS_DIR}")

def load_commodity_data(commodity_name):
    """Load CSV data for a commodity - BarChart format (no headers)"""
    csv_files = {
        'wheat': 'wheat.csv',
        'milling_wheat': 'millingwheat.csv',
        'palm': 'palmoil.csv',
        'sugar': 'sugar.csv',
        'aluminum': 'alumnium.csv',
        'crude_palm': 'brentcrude.csv'
    }
    
    if commodity_name not in csv_files:
        print(f"Commodity '{commodity_name}' not supported")
        return None
    
    csv_filename = csv_files[commodity_name]
    
    # Search for CSV file
    possible_paths = [
        os.path.join(BASE_DIR, 'data', csv_filename),
        os.path.join(BASE_DIR, csv_filename),
        os.path.join(os.path.dirname(BASE_DIR), 'data', csv_filename),
        csv_filename
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"Found {commodity_name} data at: {csv_path}")
            break
    
    if csv_path is None:
        print(f"CSV file not found for {commodity_name}. Tried paths: {possible_paths}")
        return None
    
    try:
        # Load CSV WITHOUT HEADERS
        print(f"Loading {commodity_name} without headers...")
        df = pd.read_csv(csv_path, header=None)
        
        # BarChart format: Symbol, Date, Open, High, Low, Close, Volume, (Open Interest)
        if df.shape[1] >= 7:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'] + [f'extra_{i}' for i in range(7, df.shape[1])]
        elif df.shape[1] >= 6:
            df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close'] + [f'extra_{i}' for i in range(6, df.shape[1])]
        else:
            print(f"CSV has only {df.shape[1]} columns, need at least 6")
            return None
        
        print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only necessary columns
        df = df[['date', 'close']].dropna()
        
        if len(df) == 0:
            print(f"No valid data after cleaning for {commodity_name}")
            return None
            
        print(f"Cleaned data: {len(df)} rows for {commodity_name}")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
            
    except Exception as e:
        print(f"Error loading {commodity_name} from {csv_path}: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def create_features(df, forecast_horizon=12):
    """Create features for ML model"""
    if df is None or len(df) < 30:
        print("Not enough data for feature engineering")
        return None
    
    df_features = df.copy()
    
    # Basic date features
    df_features['day'] = df_features['date'].dt.day
    df_features['month'] = df_features['date'].dt.month
    df_features['year'] = df_features['date'].dt.year
    df_features['day_of_week'] = df_features['date'].dt.dayofweek
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
    
    # Lag features
    max_lag = min(20, len(df) - 1)
    for lag in [1, 2, 3, 5, 7, 10, 14, 20]:
        if lag <= max_lag:
            df_features[f'lag_{lag}'] = df_features['close'].shift(lag)
    
    # Rolling statistics
    window_7 = min(7, len(df) // 4)
    window_30 = min(30, len(df) // 2)
    
    if window_7 > 1:
        df_features['rolling_mean_7'] = df_features['close'].rolling(window=window_7).mean()
        df_features['rolling_std_7'] = df_features['close'].rolling(window=window_7).std()
    
    if window_30 > 1:
        df_features['rolling_mean_30'] = df_features['close'].rolling(window=window_30).mean()
        df_features['rolling_min_30'] = df_features['close'].rolling(window=window_30).min()
        df_features['rolling_max_30'] = df_features['close'].rolling(window=window_30).max()
    
    # Price changes
    df_features['price_change_1'] = df_features['close'].pct_change(periods=1)
    if len(df) > 7:
        df_features['price_change_7'] = df_features['close'].pct_change(periods=min(7, len(df)-1))
    
    # Seasonal features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)
    
    # Drop rows with NaN values
    df_features = df_features.dropna()
    
    if len(df_features) < 10:
        print(f"Not enough data after feature engineering: {len(df_features)} rows")
        return None
    
    print(f"Created features with {len(df_features)} rows")
    return df_features

def train_forecast_model(commodity_name, forecast_months=12):
    """Train ML model and generate forecasts"""
    print(f"\n=== Training forecast model for {commodity_name} ===")
    
    df = load_commodity_data(commodity_name)
    if df is None:
        return None, f"Failed to load data for {commodity_name}"
    
    if len(df) < 50:
        return None, f"Insufficient data for {commodity_name}: only {len(df)} rows (need at least 50)"
    
    print(f"Loaded {len(df)} rows of data for {commodity_name}")
    
    # Prepare data
    df_features = create_features(df)
    if df_features is None:
        return None, "Failed to create features"
    
    if len(df_features) < 30:
        return None, f"Not enough data after feature engineering: {len(df_features)} rows"
    
    # Split data
    train_size = max(30, int(len(df_features) * 0.8))
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:] if len(df_features) > train_size + 10 else None
    
    # Feature columns
    feature_cols = [col for col in train_df.columns if col not in ['date', 'close']]
    
    if len(feature_cols) == 0:
        return None, "No features created for model training"
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['close']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model - DISABLE PARALLEL PROCESSING to avoid joblib issues
    n_estimators = min(50, len(X_train) // 2)  # Reduced for speed
    max_depth = min(8, len(X_train) // 10)
    
    print(f"Training RandomForest with n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Key fix: Use n_jobs=1 to avoid joblib parallel issues
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1  # Changed from -1 to 1 to avoid parallel processing
    )
    
    model.fit(X_train_scaled, y_train)
    print("Model training completed")
    
    # Evaluate on test set if available
    test_predictions = None
    mape = None
    mae = None
    
    if test_df is not None and len(test_df) > 5:
        X_test = test_df[feature_cols]
        y_test = test_df['close']
        X_test_scaled = scaler.transform(X_test)
        test_predictions = model.predict(X_test_scaled)
        
        from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
        mape = mean_absolute_percentage_error(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
        print(f"Test MAPE: {mape*100:.2f}%, MAE: {mae:.4f}")
    
    # Generate future forecasts
    last_date = df['date'].max()
    future_dates = []
    future_predictions = []
    
    # Prepare last known data for forecasting
    last_row = df_features.iloc[-1]
    current_features = last_row[feature_cols].values.reshape(1, -1)
    
    print(f"Generating {forecast_months}-month forecast from {last_date.date()}")
    
    for i in range(forecast_months):
        future_date = last_date + timedelta(days=30 * (i + 1))
        future_features = current_features.copy()
        
        # Update temporal features
        if 'month' in feature_cols:
            future_features[0, feature_cols.index('month')] = future_date.month
        if 'year' in feature_cols:
            future_features[0, feature_cols.index('year')] = future_date.year
        if 'month_sin' in feature_cols:
            future_features[0, feature_cols.index('month_sin')] = np.sin(2 * np.pi * future_date.month/12)
        if 'month_cos' in feature_cols:
            future_features[0, feature_cols.index('month_cos')] = np.cos(2 * np.pi * future_date.month/12)
        
        # Update day features
        if 'day' in feature_cols:
            future_features[0, feature_cols.index('day')] = future_date.day
        if 'day_of_week' in feature_cols:
            future_features[0, feature_cols.index('day_of_week')] = future_date.weekday()
        if 'day_of_year' in feature_cols:
            future_features[0, feature_cols.index('day_of_year')] = future_date.timetuple().tm_yday
        if 'week_of_year' in feature_cols:
            future_features[0, feature_cols.index('week_of_year')] = future_date.isocalendar().week
        
        # Scale features
        future_features_scaled = scaler.transform(future_features)
        
        # Make prediction
        prediction = model.predict(future_features_scaled)[0]
        
        # Update lag features
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
    
    print(f"Generated {len(future_predictions)} forecast points")
    
    # Save model
    try:
        model_path = os.path.join(MODELS_DIR, f'{commodity_name}_model.pkl')
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_date': last_date.strftime('%Y-%m-%d')
        }, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Warning: Could not save model: {str(e)}")
    
    # Prepare results
    historical_points = min(100, len(df))
    historical_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()[-historical_points:]
    historical_prices = df['close'].tolist()[-historical_points:]
    
    test_data = {'dates': [], 'actual': [], 'predicted': []}
    if test_df is not None and test_predictions is not None:
        test_dates = test_df['date'].dt.strftime('%Y-%m-%d').tolist()
        test_data = {
            'dates': test_dates,
            'actual': y_test.tolist() if test_df is not None else [],
            'predicted': test_predictions.tolist() if test_predictions is not None else []
        }
    
    result = {
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
    }
    
    print(f"=== Forecast completed for {commodity_name} ===")
    return result, None

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """API endpoint for forecasting"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        commodity = data.get('commodity')
        months = data.get('months', 12)
        
        print(f"Received forecast request: commodity={commodity}, months={months}")
        
        if not commodity:
            return jsonify({'error': 'Commodity parameter is required'}), 400
        
        if commodity not in ['wheat', 'milling_wheat', 'palm', 'sugar', 'aluminum', 'crude_palm']:
            return jsonify({'error': f'Unsupported commodity: {commodity}'}), 400
        
        result, error = train_forecast_model(commodity, months)
        
        if error:
            print(f"Forecast error: {error}")
            return jsonify({'error': error}), 400
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Unexpected error in forecast endpoint:\n{error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace}), 500

@app.route('/api/forecast/status', methods=['GET'])
def forecast_status():
    """Check if forecasting is available"""
    return jsonify({
        'status': 'available',
        'supported_commodities': ['wheat', 'milling_wheat', 'palm', 'sugar', 'aluminum', 'crude_palm'],
        'message': 'Machine learning forecasting API is running'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'service': 'ml_forecast_api',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Commodity ML Forecast API',
        'version': '1.0',
        'endpoints': {
            'POST /api/forecast': 'Generate ML forecasts for commodities',
            'GET /api/forecast/status': 'Check API status',
            'GET /api/health': 'Health check'
        }
    })

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')  # FIXED: Use 0.0.0.0 for Render
    
    print("=" * 60)
    print("ML Forecast API Starting...")
    print(f"Host: {host}, Port: {port}")
    print("NOTE: Using n_jobs=1 to avoid joblib compatibility issues")
    print(f"Server will run on: http://{host}:{port}")
    print("=" * 60)
    
    # FIXED: Bind to all interfaces (0.0.0.0) instead of localhost
    app.run(debug=True, host=host, port=port, use_reloader=False)