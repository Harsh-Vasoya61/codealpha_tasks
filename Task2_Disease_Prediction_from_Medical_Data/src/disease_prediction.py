"""
Task 2: Disease Prediction from Medical Data
==============================================
Objective : Predict the likelihood of disease (breast cancer diagnosis)
            using diagnostic medical measurements.
Approach  : Classification algorithms - Logistic Regression, Random Forest, SVM.
Metrics   : Accuracy, Precision, Recall, F1-Score, ROC-AUC.

Dataset: Breast Cancer Wisconsin (Diagnostic) - a REAL, well-known medical
dataset (569 patients, 30 diagnostic features derived from digitized images
of breast mass tissue). Target: diagnosis (1 = malignant/disease, 0 = benign).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")



# 1. LOAD DATA

def load_data(path="../data/breast_cancer_data.csv"):
    df = pd.read_csv(path)
    return df



# 2. TRAIN

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models



# 3. EVALUATE

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
        print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df, roc_data



# 4. PLOTS

def plot_roc_curves(roc_data, out_path):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Disease Prediction"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_confusion_matrices(models, X_test, y_test, out_path):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_feature_importance(rf_model, feature_names, out_path, top_n=15):
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 7))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="rocket", legend=False)
    plt.title("Top Feature Importances (Random Forest)"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_metrics_comparison(results_df, out_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    melted = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.05); plt.title("Model Performance Comparison")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()



# 5. PREDICT NEW PATIENT

def predict_new_patient(model, scaler, feature_columns, patient_dict):
    """Predict disease risk for a single new patient given their diagnostic measurements."""
    df_one = pd.DataFrame([patient_dict])[feature_columns]
    df_one_scaled = pd.DataFrame(scaler.transform(df_one), columns=feature_columns)
    pred_class = model.predict(df_one_scaled)[0]
    pred_proba = model.predict_proba(df_one_scaled)[0, 1]
    return pred_class, round(pred_proba, 4)



# 6. MAIN

def main():
    import os
    os.makedirs("../results", exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Disease rate (malignant): {df['diagnosis'].mean():.2%}")

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("\nTraining models: Logistic Regression, Random Forest, SVM...")
    models = train_models(X_train_s, y_train)

    print("\nEvaluating...")
    results_df, roc_data = evaluate_models(models, X_test_s, y_test)

    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    print(results_df.to_string(index=False))
    results_df.to_csv("../results/model_comparison_results.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    predictions_df = X_test.copy()
    predictions_df["actual_diagnosis"] = y_test.values
    predictions_df["predicted_diagnosis"] = best_model.predict(X_test_s)
    predictions_df["predicted_probability"] = best_model.predict_proba(X_test_s)[:, 1].round(4)
    predictions_df.to_csv("../results/predictions.csv", index=False)
    print(f"\nSaved predictions using best model: {best_model_name}")

    print("\nGenerating plots...")
    plot_roc_curves(roc_data, "../results/roc_curves.png")
    plot_confusion_matrices(models, X_test_s, y_test, "../results/confusion_matrices.png")
    plot_feature_importance(models["Random Forest"], X.columns, "../results/feature_importance.png")
    plot_metrics_comparison(results_df, "../results/metrics_comparison.png")

    print(f"\nBest model by ROC-AUC: {best_model_name}")

    # Save model
    import joblib
    os.makedirs("../models", exist_ok=True)
    joblib.dump(
        {"model": best_model, "scaler": scaler, "feature_columns": list(X.columns), "model_name": best_model_name},
        "../models/best_model.pkl",
    )
    print("Saved trained model to ../models/best_model.pkl")

    return models, best_model_name, scaler, list(X.columns)


if __name__ == "__main__":
    models, best_model_name, scaler, feature_columns = main()
    best_model = models[best_model_name]

    # Demo: predict a new patient (using an averaged "borderline" profile)
    sample_patient = {col: 15.0 for col in feature_columns}  # placeholder demo values
    pred_class, pred_proba = predict_new_patient(best_model, scaler, feature_columns, sample_patient)
    print(f"\n{'=' * 55}\nDEMO: New Patient Prediction ({best_model_name})\n{'=' * 55}")
    print(f"Predicted diagnosis : {'MALIGNANT (disease)' if pred_class == 1 else 'BENIGN'}")
    print(f"Malignancy probability : {pred_proba:.2%}")
