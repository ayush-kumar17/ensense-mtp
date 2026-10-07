import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # 1. Configuration & Paths
    data_path = "models/dataset/diabetes/diabetes_train.csv"  # Update if your dataset path differs
    target_col = "label"             # Update to match your dataset target column name
    output_dir = "models/outputs"
    output_path = os.path.join(output_dir, "stump_ensemble.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please verify your file path.")

    # 2. Load and Prepare Data
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train XGBoost Model with Strict Depth-1 Constraint
    print("Training depth-1 XGBoost ensemble (100 stumps)...")
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=1,            # CRITICAL: Forces depth-1 (decision stumps)
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluate Accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Test Accuracy: {acc:.4f}")

    # 5. Save Model as JSON (Ensense-Compatible)
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(output_path)
    print(f"Model successfully saved to {output_path}")

if __name__ == "__main__":
    main()