from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Cloud SDK: boto3 (AWS)
import boto3
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường (được đặt trong systemd service)
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    # TODO 2.6.1: Tạo một boto3 client
    # TODO 2.6.2: Lấy bucket bằng client.bucket(S3_BUCKET) - actually boto3 uses resource
    # TODO 2.6.3: Tải file xuống bằng client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    # TODO 2.6.4: In thông báo thành công
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'test'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'test')
    )
    s3_client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Model downloaded from s3://{S3_BUCKET}/{S3_MODEL_KEY} to {MODEL_PATH}")


# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server. GitHub Actions dùng endpoint này để xác nhận deploy thành công."""
    # TODO 2.6.6: Trả về dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.

    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra:  JSON {"prediction": <0|1|2>, "label": <"thấp"|"trung_bình"|"cao">}
    """
    # TODO 2.6.7: Kiểm tra len(req.features) == 12.
    #   Nếu không, raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # TODO 2.6.8: Gọi model.predict([req.features]) để lấy kết quả dự đoán.
    prediction = model.predict([req.features])[0]

    # TODO 2.6.9: Trả về dict chứa "prediction" (int) và "label" (string).
    #   Nhãn: 0 -> "thấp", 1 -> "trung_bình", 2 -> "cao"
    labels = ["thấp", "trung_bình", "cao"]
    return {
        "prediction": int(prediction),
        "label": labels[int(prediction)]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)