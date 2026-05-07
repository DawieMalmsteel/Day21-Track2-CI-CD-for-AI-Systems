import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# mlflow may not be importable in minimal test environments; provide a no-op fallback.
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    class _MLFlowSklearnStub:
        @staticmethod
        def log_model(model, name):
            pass

    class _MLFlowStub:
        def start_run(self):
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                yield None
            return _cm()

        def log_params(self, params):
            pass

        def log_metric(self, key, value):
            pass

    mlflow = _MLFlowStub()
    mlflow.sklearn = _MLFlowSklearnStub()

EVAL_THRESHOLD = 0.60


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá
    """

    # Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # Tách đặc trưng và nhãn
    if "target" not in df_train.columns or "target" not in df_eval.columns:
        raise ValueError("Input CSV files must contain a 'target' column")

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        # Ghi nhận siêu tham số
        mlflow.log_params(params)

        # Khởi tạo và huấn luyện mô hình
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Dự đoán và tính chỉ số
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # Ghi metrics và model vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # In kết quả
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Lưu metrics ra file
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # Lưu mô hình ra file
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
