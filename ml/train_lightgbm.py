from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


FEATURES = ["return", "range", "body", "atr"]


def train(csv_path: str, model_path: str = "models/lightgbm_model.pkl") -> None:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is not installed")

    df = pd.read_csv(csv_path)
    df["future_return"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.dropna()

    x = df[FEATURES]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print(classification_report(y_test, preds))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    train("data/history.csv")
