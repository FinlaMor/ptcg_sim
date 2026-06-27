import os
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration Parameters
DATA_PATH = "upgraded_training_data.csv"
MODEL_OUTPUT = "rotom_computer.txt"
PLOT_OUTPUT = "feature_importance.png"

def plot_importance(model, feature_names, max_features=15):
    """Calculates and exports a clean visualization of feature importances."""
    importances = model.feature_importance(importance_type='split')
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True).tail(max_features)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue', edgecolor='black', height=0.6)
    plt.title(f"Top {max_features} Most Influential PTCG Game State Features", fontsize=14, pad=15)
    plt.xlabel("Split Count", fontsize=11, labelpad=10)
    plt.ylabel("Game Feature", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300)
    plt.close()

def train_upgraded_model():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training source '{DATA_PATH}' not found.")
        return 1

    df = pd.read_csv(DATA_PATH)
    if len(df) < 50:
        print("[NOTICE] Not enough data points to reliably gate yet. Skipping update.")
        return 0

    X = df.drop(columns=['label'])
    y = df['label']

    # Partition Data into Train and Test Splits (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- BASELINE VALUATION OF CURRENT DEPLOYED BRAIN ---
    current_baseline_rmse = float('inf')
    if os.path.exists(MODEL_OUTPUT):
        try:
            with open(MODEL_OUTPUT, 'r', encoding='utf-8') as f:
                old_model_str = f.read()
            old_bst = lgb.Booster(model_str=old_model_str)
            old_preds = old_bst.predict(X_test)
            current_baseline_rmse = np.sqrt(mean_squared_error(y_test, old_preds))
            print(f"| Current Deployed Model Baseline RMSE on New Data: {current_baseline_rmse:.4f}")
        except Exception:
            print("| No valid old model baseline could be evaluated. Defaulting gate to open.")

    # Construct LightGBM Native Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 24,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42
    }

    print("Beginning training iterations...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
    ]
    
    new_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        callbacks=callbacks
    )

    # New Model Evaluation
    predictions = new_model.predict(X_test, num_iteration=new_model.best_iteration)
    new_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"| Candidate Model RMSE: {new_rmse:.4f}")
    print(f"| Candidate Model R² Score: {r2:.4f}")

    # --- THE GATE CHECK ---
    # We only overwrite the live model if the candidate's error is strictly lower
    if new_rmse < current_baseline_rmse:
        print("\n[GATE PASSED] New model out-performed the baseline! Deploying...")
        plot_importance(new_model, feature_names=X.columns.tolist())
        model_string = new_model.model_to_string()
        with open(MODEL_OUTPUT, "w", encoding="utf-8") as f:
            f.write(model_string)
        print("Live model successfully updated to new generation.")
    else:
        print("\n[GATE REJECTED] Candidate model regressed or failed to beat baseline.")
        print("Live model footprint unchanged. Retaining previous generation brain.")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(train_upgraded_model())
