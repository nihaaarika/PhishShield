from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from scamguard.model import build_model_pipeline, prepare_texts


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ScamGuard model (TF-IDF + Logistic Regression).")
    parser.add_argument("--data", default="data/sample_messages.csv", help="CSV with columns: text,label")
    parser.add_argument("--out", default="scamguard/artifacts/model.joblib", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = prepare_texts(df["text"].astype(str).tolist())
    y = df["label"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, out_path)
    print(f"Saved model -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

