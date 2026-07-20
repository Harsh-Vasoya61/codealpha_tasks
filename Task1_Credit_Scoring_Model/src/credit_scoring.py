"""
Credit Scoring Model — Realistic Synthetic Dataset (v2)
==========================================================
Uses credit_data_realistic.csv (20,000 rows, ~26% default rate,
missing values, categorical + numeric features, non-linear risk drivers).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# 1. LOAD + CLEAN
def load_and_clean(path="../data/credit_data.csv"):
    df = pd.read_csv(path)

    # Impute missing numeric values with median
    for col in ["annual_income", "employment_length_years"]:
        df[col] = SimpleImputer(strategy="median").fit_transform(df[[col]])

    return df


# 2. FEATURE ENGINEERING
def engineer_features(df):
    df = df.copy()
    df["debt_to_income"] = df["existing_debt"] / (df["annual_income"] + 1)
    df["loan_to_income"] = df["loan_amount_requested"] / (df["annual_income"] + 1)
    df["credit_used_amount"] = df["credit_limit"] * df["credit_utilization"]
    df["income_per_account"] = df["annual_income"] / (df["num_open_accounts"] + 1)
    df["late_payment_rate"] = df["num_late_payments_12m"] / (df["credit_history_length"] + 1)
    df["high_risk_combo"] = (
        (df["credit_utilization"] > 0.7) & (df["num_late_payments_12m"] > 1)
    ).astype(int)

    df = pd.get_dummies(
        df,
        columns=["education", "marital_status", "employment_type", "home_ownership", "purpose"],
        drop_first=True,
    )
    return df


# 3. TRAIN
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=15,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


# 4. EVALUATE
def evaluate_models(models, X_test, y_test):
    results, roc_data = [], {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_proba))

        print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
        print(classification_report(y_test, y_pred, target_names=["Good (0)", "Default (1)"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df, roc_data


# 5. PLOTS
def plot_roc_curves(roc_data, out_path):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Realistic Dataset"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_confusion_matrices(models, X_test, y_test, out_path):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Good", "Default"], yticklabels=["Good", "Default"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_feature_importance(rf_model, feature_names, out_path, top_n=15):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 7))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False)
    plt.title("Top Feature Importances (Random Forest)"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_metrics_comparison(results_df, out_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    melted = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1); plt.title("Model Performance Comparison")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# 6. MAIN
def main():
    print("Loading realistic dataset...")
    df = load_and_clean()
    print(f"Shape: {df.shape} | Default rate: {df['default'].mean():.2%}")

    df_fe = engineer_features(df)
    X = df_fe.drop(columns=["default"])
    y = df_fe["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("\nTraining models...")
    models = train_models(X_train_s, y_train)

    print("\nEvaluating...")
    results_df, roc_data = evaluate_models(models, X_test_s, y_test)

    import os
    os.makedirs("../results", exist_ok=True)

    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    print(results_df.to_string(index=False))
    results_df.to_csv("../results/model_comparison_results.csv", index=False)

    # --- Save actual predictions (not just metrics) ---
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    predictions_df = X_test.copy()
    predictions_df["actual_default"] = y_test.values
    predictions_df["predicted_default"] = best_model.predict(X_test_s)
    predictions_df["predicted_probability"] = best_model.predict_proba(X_test_s)[:, 1].round(4)
    predictions_df.to_csv("../results/predictions.csv", index=False)
    print(f"\nSaved {len(predictions_df)} row-level predictions to predictions_realistic.csv "
          f"(using best model: {best_model_name})")
    print("\nSample predictions:")
    print(predictions_df[["actual_default", "predicted_default", "predicted_probability"]].head(10).to_string(index=False))

    print("\nGenerating plots...")
    plot_roc_curves(roc_data, "../results/roc_curves.png")
    plot_confusion_matrices(models, X_test_s, y_test, "../results/confusion_matrices.png")
    plot_feature_importance(models["Random Forest"], X.columns, "../results/feature_importance.png")
    plot_metrics_comparison(results_df, "../results/metrics_comparison.png")

    best = results_df.iloc[0]["Model"]
    print(f"\nBest model by ROC-AUC: {best}")

    # --- Save best model + scaler + feature list for reuse ---
    import joblib, os
    os.makedirs("../models", exist_ok=True)
    joblib.dump(
        {"model": models[best], "scaler": scaler, "feature_columns": list(X.columns), "model_name": best},
        "../models/best_model.pkl",
    )
    print("Saved trained model to ../models/best_model.pkl")

    return models, best_model_name, scaler, list(X.columns)


def predict_new_applicant(model, scaler, feature_columns, applicant_dict):
    """
    Predict default risk for a single new applicant.

    applicant_dict: dict of raw feature values, e.g.
        {
            "age": 35, "annual_income": 55000, "employment_length_years": 6,
            "credit_history_length": 10, "num_open_accounts": 4,
            "credit_limit": 15000, "credit_utilization": 0.4,
            "existing_debt": 8000, "num_late_payments_12m": 1,
            "num_credit_inquiries_6m": 2, "loan_amount_requested": 10000,
            "education": "bachelors", "marital_status": "married",
            "employment_type": "salaried", "home_ownership": "MORTGAGE",
            "purpose": "debt_consolidation",
        }
    Returns: (predicted_class, probability_of_default)
    """
    df_one = pd.DataFrame([applicant_dict])
    df_one = engineer_features(pd.concat([df_one, pd.DataFrame({"default": [0]})], axis=1))
    df_one = df_one.drop(columns=["default"])

    # Align columns with training features (missing dummy columns -> 0)
    for col in feature_columns:
        if col not in df_one.columns:
            df_one[col] = 0
    df_one = df_one[feature_columns]

    df_one_scaled = pd.DataFrame(scaler.transform(df_one), columns=feature_columns)
    pred_class = model.predict(df_one_scaled)[0]
    pred_proba = model.predict_proba(df_one_scaled)[0, 1]
    return pred_class, round(pred_proba, 4)


if __name__ == "__main__":
    models, best_model_name, scaler, feature_columns = main()
    best_model = models[best_model_name]

    # --- Demo: predict a single new applicant ---
    sample_applicant = {
        "age": 29, "annual_income": 42000, "employment_length_years": 3,
        "credit_history_length": 6, "num_open_accounts": 5,
        "credit_limit": 9000, "credit_utilization": 0.78,
        "existing_debt": 11000, "num_late_payments_12m": 2,
        "num_credit_inquiries_6m": 3, "loan_amount_requested": 8000,
        "education": "bachelors", "marital_status": "single",
        "employment_type": "salaried", "home_ownership": "RENT",
        "purpose": "credit_card",
    }
    pred_class, pred_proba = predict_new_applicant(best_model, scaler, feature_columns, sample_applicant)
    print(f"\n{'=' * 55}\nDEMO: New Applicant Prediction ({best_model_name})\n{'=' * 55}")
    print(f"Predicted class     : {'DEFAULT RISK' if pred_class == 1 else 'GOOD CREDIT'}")
    print(f"Default probability : {pred_proba:.2%}")
