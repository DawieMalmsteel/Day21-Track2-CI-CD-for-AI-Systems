import os
# Set MLflow tracking URI and artifact root from environment variables
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
mlflow_artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
if mlflow_artifact_root:
    os.environ["MLFLOW_ARTIFACT_ROOT"] = mlflow_artifact_root

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


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
    # TODO 1.5.1: Đọc dữ liệu huấn luyện từ data_path vào DataFrame df_train
    #   và dữ liệu đánh giá từ eval_path vào DataFrame df_eval.
    # Gợi ý: sử dụng pd.read_csv(...)
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 1.5.2: Tách đặc trưng và nhãn.
    #   X_train, y_train từ df_train (bỏ cột "target")
    #   X_eval, y_eval từ df_eval (bỏ cột "target")
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # TODO 1.5.3: Bắt đầu một MLflow run bằng `with mlflow.start_run():`
    #   Bên trong block này, thực hiện các bước sau:
    with mlflow.start_run():
        # TODO 1.5.4: Ghi nhận các siêu tham số vào MLflow.
        #   Gợi ý: mlflow.log_params(params)
        mlflow.log_params(params)

        # TODO 1.5.5: Khởi tạo và huấn luyện mô hình RandomForestClassifier.
        #   Gợi ý: model = RandomForestClassifier(**params, random_state=42)
        #          model.fit(X_train, y_train)
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 1.5.6: Tính accuracy và f1_score trên tập đánh giá.
        #   Gợi ý: preds = model.predict(X_eval)
        #          acc = accuracy_score(y_eval, preds)
        #          f1  = f1_score(y_eval, preds, average="weighted")
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 1.5.7: Ghi nhận các chỉ số vào MLflow.
        #   Gợi ý: mlflow.log_metric("accuracy", acc)
        #          mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # TODO 1.5.8: Log mô hình vào MLflow artifact.
        #   Gợi ý: mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(model, "model")

        # TODO 1.5.9: In kết quả ra màn hình.
        #   Gợi ý: print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 1.5.10: Lưu metrics ra file outputs/metrics.json.
        #   File này sẽ được đọc bởi GitHub Actions ở Bước 2.
        #   Gợi ý:
        #       os.makedirs("outputs", exist_ok=True)
        #       with open("outputs/metrics.json", "w") as f:
        #           json.dump({"accuracy": acc, "f1_score": f1}, f)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 1.5.11: Lưu mô hình ra file models/model.pkl.
        #   File này sẽ được upload lên GCS ở Bước 2.
        #   Gợi ý:
        #       os.makedirs("models", exist_ok=True)
        #       joblib.dump(model, "models/model.pkl")
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # TODO 1.5.12: Trả về acc để các hàm gọi train() có thể đọc kết quả.
        return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)