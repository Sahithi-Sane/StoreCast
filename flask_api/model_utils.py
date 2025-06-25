import boto3
import joblib
import pandas as pd
import io
import os
import pickle
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
bucket = os.getenv('AWS_BUCKET_NAME')

def load_model_from_s3(store_id):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    models = {}
    try:
        model_dir = f'models/store_{store_id}/'
        
        # Load AutoARIMA model
        arima_model_file = s3.get_object(Bucket=bucket, Key=f'{model_dir}store_{store_id}_auto_arima.pkl')
        models['auto_arima'] = joblib.load(io.BytesIO(arima_model_file['Body'].read()))

        # Load XGBoost model
        xgb_model_file = s3.get_object(Bucket=bucket, Key=f'{model_dir}store_{store_id}_xgb_model.pkl')
        models['xgb'] = joblib.load(io.BytesIO(xgb_model_file['Body'].read()))

    except Exception as e:
        print(f"Error loading models for Store {store_id}: {e}")
        return None, None  # Handle error if models don't exist
    
    return models['auto_arima'], models['xgb']

def load_data_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    obj = s3.get_object(Bucket=bucket, Key='data/clean_data.csv')
    return pd.read_csv(io.BytesIO(obj['Body'].read()))
