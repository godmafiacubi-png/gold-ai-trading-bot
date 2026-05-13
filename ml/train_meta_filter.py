from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from ml.meta_features import MIN_META_DATASET_ROWS, META_FEATURE_COLUMNS


def main():
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm not installed. Run: pip install lightgbm")

    df = pd.read_csv("data/meta_dataset.csv")

    if len(df) < MIN_META_DATASET_ROWS:
        raise RuntimeError(
            f"Meta dataset too small ({len(df)} rows). Need at least "
            f"{MIN_META_DATASET_ROWS} rows."
        )

    missing = [col for col in META_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Meta dataset is missing feature columns: {missing}")

    x = df[META_FEATURE_COLUMNS]
    y = df["target"]

    split = int(len(df) * 0.7)

    x_train = x.iloc[:split]
    y_train = y.iloc[:split]

    x_test = x.iloc[split:]
    y_test = y.iloc[split:]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.55).astype(int)

    print("\n===== META FILTER REPORT =====")
    print(classification_report(y_test, pred))

    try:
        auc = roc_auc_score(y_test, proba)
        print("AUC:", round(auc, 3))
    except Exception:
        pass

    importance = pd.DataFrame({
        "feature": x.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n===== FEATURE IMPORTANCE =====")
    print(importance.to_string(index=False))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/meta_filter.pkl")
    print("\nSaved models/meta_filter.pkl")


if __name__ == "__main__":
    main()