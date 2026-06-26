import csv
import lightgbm as lgb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

INPUT_CSV = Path("training_data.csv")
MODEL_PATH = Path("pok_model.txt")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        raise ValueError("No rows found in training CSV.")

    # Select only numeric feature columns, excluding nonnumeric metadata like file names.
    feature_names = []
    for name in reader.fieldnames:
        if name == "label":
            continue
        try:
            float(rows[0][name])
            feature_names.append(name)
        except (TypeError, ValueError):
            continue

    if not feature_names:
        raise ValueError("No numeric feature columns found in training CSV.")

    X = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([int(row["label"]) for row in rows], dtype=np.int32)
    return X, y, feature_names


def main():
    X, y, feature_names = load_csv(INPUT_CSV)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "seed": RANDOM_STATE,
    }

    bst = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
    )

    y_pred = bst.predict(X_val)
    val_loss = log_loss(y_val, y_pred)
    val_acc = accuracy_score(y_val, [1 if p >= 0.5 else 0 for p in y_pred])

    bst.save_model(str(MODEL_PATH))

    print(f"Trained model saved to {MODEL_PATH}")
    print(f"Validation binary_logloss: {val_loss:.5f}")
    print(f"Validation accuracy: {val_acc:.5f}")


if __name__ == "__main__":
    main()
