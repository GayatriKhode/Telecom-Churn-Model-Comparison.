"""Compare Logistic Regression and Decision Tree for telecom churn.
Run: python compare_models.py --data "data/telecom_churn.csv"
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent

def encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "telecom_churn.csv"))
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.data)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        sys.exit(f"ERROR: Dataset CSV not found: {path}")

    df = pd.read_csv(path)
    target_matches = [c for c in df.columns if str(c).strip().lower() == args.target.lower()]
    if not target_matches:
        target_matches = [c for c in df.columns if str(c).strip().lower() in {"churn", "exited", "target", "churn_label"}]
    if not target_matches:
        sys.exit("ERROR: Target column not found. Available columns: " + ", ".join(map(str, df.columns)))

    target_col = target_matches[0]
    df = df.dropna(subset=[target_col]).copy()
    y_raw = df.pop(target_col).astype(str).str.strip()
    low = y_raw.str.lower()
    yes_no = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0,
              "churn": 1, "not churn": 0, "exited": 1, "not exited": 0}
    if set(low.unique()).issubset(yes_no):
        y = low.map(yes_no).astype(int)
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw), index=y_raw.index)
        if len(le.classes_) != 2:
            sys.exit("ERROR: The target column must have exactly two classes.")

    ids = [c for c in df.columns if str(c).strip().lower() in {"customerid", "customer_id", "id", "rowid"}]
    X = df.drop(columns=ids) if ids else df.copy()
    for col in X.columns:
        if X[col].dtype == "object":
            cleaned = X[col].astype(str).str.strip().replace({"": None, "nan": None})
            numeric = pd.to_numeric(cleaned, errors="coerce")
            valid = cleaned.notna().sum()
            if valid and numeric.notna().sum() >= 0.8 * valid:
                X[col] = numeric

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", encoder())])
    logistic_prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num_cols),
        ("categorical", cat_pipe, cat_cols)])
    tree_prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("categorical", cat_pipe, cat_cols)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessing", logistic_prep),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([
            ("preprocessing", tree_prep),
            ("classifier", DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                                  random_state=args.random_state,
                                                  class_weight="balanced"))])
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })

    results = pd.DataFrame(rows).set_index("Model")
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "model_comparison.csv"
    results.to_csv(csv_path, float_format="%.4f")
    winner = results["F1-score"].idxmax()
    conclusion = (f"{winner} performed better based on F1-score "
                  f"({results.loc[winner, 'F1-score']:.4f}) on this test split. "
                  "Review recall and ROC-AUC as well before selecting a final model.")
    (out / "conclusion.txt").write_text(conclusion + "\n", encoding="utf-8")
    print("\nMODEL COMPARISON RESULTS")
    print(results.round(4).to_string())
    print(f"\nSaved comparison table: {csv_path}")
    print(f"Saved conclusion: {out / 'conclusion.txt'}")
    print("\nConclusion:", conclusion)

if __name__ == "__main__":
    main()
