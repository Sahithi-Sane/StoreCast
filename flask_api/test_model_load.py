import boto3, os, joblib, io
import traceback

store_id = "1"  # Change if needed

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bucket = os.getenv("AWS_BUCKET_NAME")
model_dir = f"models/store_{store_id}/"

try:
    # Load AutoARIMA model
    arima_key = f"{model_dir}store_{store_id}_auto_arima.pkl"
    print(f"📄 Trying to load: {arima_key}")
    arima_response = s3.get_object(Bucket=bucket, Key=arima_key)
    arima_model = joblib.load(io.BytesIO(arima_response['Body'].read()))
    print("✅ AutoARIMA model loaded successfully")

    # Load XGBoost model
    xgb_key = f"{model_dir}store_{store_id}_xgb_model.pkl"
    print(f"📄 Trying to load: {xgb_key}")
    xgb_response = s3.get_object(Bucket=bucket, Key=xgb_key)
    xgb_model = joblib.load(io.BytesIO(xgb_response['Body'].read()))
    print("✅ XGBoost model loaded successfully")

except Exception as e:
    traceback.print_exc()
    print(f"❌ Error occurred while loading models for store {store_id}: {e}")
