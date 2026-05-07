import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Wine Quality để sử dụng trong test.

    Trả về (train_path, eval_path) là đường dẫn tới file CSV.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_df = df.iloc[:160]
    eval_df = df.iloc[160:]

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực trong [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiểm tra file outputs/metrics.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # remove any existing outputs to avoid false positives
    if os.path.exists("outputs/metrics.json"):
        os.remove("outputs/metrics.json")

    train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics and "f1_score" in metrics


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.pkl được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # remove existing model if present
    if os.path.exists("models/model.pkl"):
        os.remove("models/model.pkl")

    train({"n_estimators": 10, "max_depth": 3}, data_path=train_path, eval_path=eval_path)

    assert os.path.exists("models/model.pkl")
