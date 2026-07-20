# Disease Prediction from Medical Data

Predicting breast cancer diagnosis (malignant vs. benign) using diagnostic medical measurements,
built with classification algorithms (Logistic Regression, Random Forest, SVM).

---

## 📁 Project Structure

```
Task2_Disease_Prediction_from_Medical_Data/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── breast_cancer_data.csv       <- Breast Cancer Wisconsin (Diagnostic) dataset
│
├── notebook/
│   └── Disease_Prediction_Model.ipynb   <- Full walkthrough: EDA → modeling → evaluation
│
├── src/
│   └── disease_prediction.py        <- Reusable pipeline as a plain .py module
│
├── models/
│   └── best_model.pkl                <- Saved best model + scaler + feature list
│
└── results/
    ├── model_comparison_results.csv
    ├── predictions.csv
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    └── metrics_comparison.png
```

---

## 📊 Dataset

**Breast Cancer Wisconsin (Diagnostic)** — a real, well-known medical dataset (569 patients, 30
diagnostic features computed from digitized images of breast mass tissue: radius, texture,
perimeter, area, smoothness, compactness, concavity, symmetry, etc.).

Target column: **`diagnosis`** (1 = malignant, 0 = benign). Malignant rate: ~37%.

This dataset is clean (no missing values) and ships with scikit-learn, so the project runs
end-to-end with no external download required.

---

## 🧠 Approach

1. **EDA** — class balance, correlation heatmap, feature distributions by diagnosis
2. **Preprocessing** — feature scaling (StandardScaler); no imputation needed (data is clean)
3. **Modeling** — Logistic Regression, Random Forest, SVM (all `class_weight='balanced'`)
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrices, ROC curves
5. **Model Selection** — best model chosen by ROC-AUC (missing a malignant case is far more costly
   than a false alarm, so ranking ability across thresholds matters most)
6. **Deployment prep** — best model + scaler saved to `models/best_model.pkl`; a
   `predict_new_patient()` helper scores a new patient's measurements

---

## ▶️ How to Run

**Notebook (recommended):**
```bash
cd notebook
jupyter notebook Disease_Prediction_Model.ipynb
# Run All Cells
```

**Plain script:**
```bash
cd src
python disease_prediction.py
```

---

## 📈 Results (this run)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.974 | 1.000 | 0.929 | 0.963 | **0.997** |
| SVM | 0.982 | 0.976 | 0.976 | 0.976 | 0.995 |
| Logistic Regression | 0.974 | 0.976 | 0.952 | 0.964 | 0.995 |

All three models perform very well — expected, since this dataset is a well-studied, highly
separable benchmark in medical machine learning. **Random Forest** was selected as the best model
by ROC-AUC, achieving perfect precision on the test set (no false malignant predictions).

Top predictive features (Random Forest importance) align with known clinical indicators of
malignancy — cell radius, concavity, and area among the strongest signals.

---

## 🔧 Requirements

```
pip install -r requirements.txt
```
