from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Import cloud SDK lazily to allow local dev without provider SDK
try:
    from google.cloud import storage
except Exception:
    storage = None

app = FastAPI()

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tải file model.pkl từ GCS về máy khi server khởi động.

    Nếu không có biến môi trường/SDK, chỉ in cảnh báo. Nếu file
    đã tồn tại cục bộ, không tải lại.
    """
    # Nếu model đã tồn tại cục bộ, giữ nguyên
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}, skipping download.")
        return

    if not GCS_BUCKET or storage is None:
        print("GCS_BUCKET not set or google-cloud-storage not available; skipping download.")
        return

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Model downloaded from gs://{GCS_BUCKET}/{GCS_MODEL_KEY} to {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to download model from GCS: {e}")


# Try to download model (no-op if not configured)
download_model()

# If model missing, loading will raise and server should fail fast
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiểm tra sức khỏe server.
    GitHub Actions gọi endpoint này sau khi deploy để xác nhận server đang chạy.

    Trả về: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận chính.

    Đầu vào : JSON {"features": [f1, f2, ..., f12]}
    Đầu ra : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if not isinstance(req.features, list) or len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    try:
        pred = model.predict([req.features])
        pred_int = int(pred[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred_int, "label": labels.get(pred_int, "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
