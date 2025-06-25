from flask import Flask, render_template, request, jsonify
from model_utils import load_model_from_s3, load_data_from_s3
import json
import base64
import os
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prometheus_client import Counter, Summary, Histogram, Gauge, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
data = load_data_from_s3()

# Prometheus metrics
prediction_counter = Counter('storecast_predictions_total', 'Total predictions made')
prediction_latency = Summary('storecast_prediction_latency_seconds', 'Time taken for prediction')
store_prediction_counter = Counter('storecast_predictions_by_store', 'Predictions by store', ['store_id'])
forecast_errors = Counter('storecast_forecast_errors_total', 'Total forecast errors')
data_size_gauge = Gauge('storecast_input_data_rows', 'Number of input data rows per store')
forecast_input_features_gauge = Gauge('storecast_input_feature_columns', 'Number of input features for forecast')
request_duration_histogram = Histogram('storecast_prediction_request_duration_seconds', 'Histogram of prediction durations')


def preprocess_data(df):
    df['Log_Weekly_Sales'] = np.log1p(df['Weekly_Sales'])
    df_diff = df.copy()
    df_diff['Weekly_Sales'] = df['Weekly_Sales'].diff().dropna()
    df_diff = df_diff.dropna()
    return df, df_diff


def create_features(df):
    df['Lag_1'] = df['Weekly_Sales'].shift(1)
    df['Lag_2'] = df['Weekly_Sales'].shift(2)
    df['Lag_3'] = df['Weekly_Sales'].shift(3)
    df['Rolling_Mean'] = df['Weekly_Sales'].rolling(window=4).mean()
    df['Rolling_Std'] = df['Weekly_Sales'].rolling(window=4).std()
    return df.dropna()

@app.route('/')
def introduction():
    return render_template('introduction.html')

@app.route('/walmart-data')
def walmart_data():
    return render_template('walmart_data.html')

@app.route('/data-analysis')
def analysis():
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].astype(str)
    chart_data = df.to_dict(orient='records')
    return render_template('data_analysis.html', chart_data=json.dumps(chart_data))

@app.route('/prediction')
def prediction_page():
    return render_template('predictions.html')

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

@app.route('/predict', methods=['POST'])
@prediction_latency.time()
@request_duration_histogram.time()
def predict():
    try:
        prediction_counter.inc()
        store_id = int(request.json['store_id'])
        forecast_period = int(request.json['forecast_period'])

        if store_id < 1 or store_id > 45:
            return jsonify({'error': "Invalid store ID. Please choose a store ID between 1 and 45."})

        if forecast_period < 1 or forecast_period > 150:
            return jsonify({'error': "Invalid forecast period. Please enter a number between 1 and 150 days."})

        model_auto_arima, xgb_model = load_model_from_s3(store_id)

        if model_auto_arima is None:
            forecast_errors.inc()
            return jsonify({'error': "Models for the selected store are not available."})

        df_store = load_data_from_s3()
        df_store_filtered = df_store[df_store['Store'] == store_id]
        data_size_gauge.set(len(df_store_filtered))

        df_store_filtered['Date'] = pd.to_datetime(df_store_filtered['Date'])
        df_store_filtered.set_index('Date', inplace=True)
        df_store_filtered['IsHoliday'] = df_store_filtered['IsHoliday'].replace({True: 1, False: 0})
        df_store_filtered['IsHoliday'] = df_store_filtered['IsHoliday'].apply(lambda x: 1 if x else 0)
        df_store_filtered = df_store_filtered.drop(columns=['Store'])

        df_store_numeric = df_store_filtered.select_dtypes(include=[np.number])
        df_week, df_week_diff = preprocess_data(df_store_numeric.resample('W').mean())
        df_week_diff = create_features(df_week_diff)

        xgb_features = ['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean', 'Rolling_Std', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']
        xgb_input = df_week_diff[xgb_features].iloc[-forecast_period:]
        forecast_input_features_gauge.set(xgb_input.shape[1])

        y_pred_arima = model_auto_arima.predict(n_periods=forecast_period).tolist()
        y_pred_xgb = xgb_model.predict(xgb_input).tolist()
        ensemble_predictions = [(a + b) / 2 for a, b in zip(y_pred_arima, y_pred_xgb)]

        prediction_dates = pd.date_range(df_week_diff.index[-1] + pd.Timedelta(1, unit='D'), periods=forecast_period)

        prediction_df = pd.DataFrame({
            'Date': prediction_dates.strftime('%Y-%m-%d'),
            'ARIMA': [round(val, 2) for val in y_pred_arima],
            'XGBoost': [round(val, 2) for val in y_pred_xgb],
            'Ensemble': [round(val, 2) for val in ensemble_predictions]
        })

        store_prediction_counter.labels(store_id=store_id).inc()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prediction_dates, y=y_pred_arima, mode='lines', name='ARIMA Forecast', line=dict(color='#2196F3')))
        fig.add_trace(go.Scatter(x=prediction_dates, y=y_pred_xgb, mode='lines', name='XGBoost Forecast', line=dict(color='#EC7373')))
        fig.add_trace(go.Scatter(x=prediction_dates, y=ensemble_predictions, mode='lines+markers', name='Ensemble Forecast', line=dict(color='#4CAF50', width=3)))

        fig.update_layout(
            title=f'Store {store_id} Sales Forecast for {forecast_period} Weeks',
            xaxis_title='Date',
            yaxis_title='Weekly Sales',
            template='plotly_dark',
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            legend=dict(x=0.01, y=0.99, font=dict(size=12, color='white')),
            xaxis=dict(tickangle=45, gridcolor='#444'),
            yaxis=dict(tickformat="$,.2f", gridcolor='#444'),
            margin=dict(t=100, b=100),
            height=600,
        )

        plot_json = fig.to_json()
        table_data = prediction_df.to_dict(orient='records')

        output = io.BytesIO()
        prediction_df.to_excel(output, index=False)
        output.seek(0)
        excel_data = output.getvalue()
        excel_base64 = base64.b64encode(excel_data).decode('utf-8')

        return jsonify({
            'prediction': table_data,
            'plot_data': plot_json,
            'excel_file': excel_base64,
            'dates': prediction_dates.strftime('%Y-%m-%d').tolist()
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        forecast_errors.inc()
        return jsonify({'error': str(e)})

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

