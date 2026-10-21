# Telecom Churn Model Comparison

Compares Logistic Regression and Decision Tree using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Run
1. Put your existing dataset at `data/telecom_churn.csv`.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Run: `python compare_models.py --data "data/telecom_churn.csv"`

If the target column has a different name, add `--target "YourColumnName"`.

The script builds preprocessing pipelines, trains both models on the same stratified train/test split, prints all metrics, and saves `outputs/model_comparison.csv` and `outputs/conclusion.txt`.

Do not upload datasets containing private customer information.
