# Credit Scoring Model

Predicting an individual's **creditworthiness** using past financial data, built with classification
algorithms (Logistic Regression, Decision Tree, Random Forest).

This project follows the standard task brief: *"Predict an individual's creditworthiness using past
financial data. Use classification algorithms like Logistic Regression, Decision Trees, or Random Forest.
Engineer features from financial history. Assess model accuracy using Precision, Recall, F1-Score, ROC-AUC."*

---

## 📁 Project Structure

```
CodeAlpha_Credit_Scoring_Model/
│
├── README.md                  <- You are here
├── requirements.txt            <- Python dependencies
│
├── data/
│   └── credit_data.csv         <- Dataset (20,000 rows, synthetic but realistic)
│
├── notebook/
│   └── Credit_Scoring_Model.ipynb   <- Full walkthrough: EDA → cleaning → features → models → evaluation
│
├── src/
│   ├── generate_data.py        <- Regenerates the dataset from scratch (optional)
│   └── credit_scoring.py       <- Reusable pipeline as a plain .py module (importable or runnable directly)
│
├── models/
│   └── best_model.pkl          <- Saved best model + scaler + feature list (created after running the notebook)
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

`data/credit_data.csv` — 20,000 rows, 17 columns, **~26% default rate** (realistic imbalance for credit
portfolios). Includes demographics (age, education, marital status, employment type), financial features
(income, existing debt, credit limit, credit utilization), payment history (late payments, credit
inquiries), loan request details, and ~4% missing values in a couple of columns (like real bureau data).

Target column: **`default`** (1 = defaulted / bad credit risk, 0 = good credit).

Want your own data instead? Swap `data/credit_data.csv` for any CSV with a binary `default` column —
the pipeline works as-is.

---

## 🧠 Approach

1. **EDA** — class balance, missing values, distributions, correlation heatmap
2. **Data Cleaning** — median imputation for missing income/employment length
3. **Feature Engineering** — debt-to-income ratio, loan-to-income ratio, credit amount used,
   income per account, late-payment rate, high-risk interaction flag
4. **Modeling** — Logistic Regression, Decision Tree, Random Forest (all `class_weight='balanced'`
   to handle the imbalanced target)
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrices, ROC curves,
   feature importance
6. **Model Selection** — best model chosen by ROC-AUC (most informative metric for imbalanced
   credit-risk classification)
7. **Deployment prep** — best model + scaler saved to `models/best_model.pkl`; a `predict_new_applicant()`
   helper scores a brand-new applicant

---

## ▶️ How to Run

**Option A — Notebook (recommended, matches the walkthrough above):**
```bash
cd notebook
jupyter notebook Credit_Scoring_Model.ipynb
# Run All Cells
```

**Option B — Plain script:**
```bash
cd src
python credit_scoring.py
```

Both produce the same results, saved to `results/` and `models/`.

---

## 📈 Results (this run)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.732 | 0.492 | 0.697 | 0.577 | **0.794** |
| Random Forest | 0.740 | 0.504 | 0.632 | 0.561 | 0.786 |
| Decision Tree | 0.711 | 0.465 | 0.670 | 0.549 | 0.766 |

**Logistic Regression** was selected as the best model — highest ROC-AUC, meaning it best separates
high-risk from low-risk applicants across all decision thresholds (more informative here than raw
accuracy, since missing a risky applicant is more costly than a false alarm).

Top predictive features (Random Forest importance): **credit utilization**, **debt-to-income ratio**,
and **late payment history** — consistent with real-world credit risk drivers.

---

## 🔧 Requirements

```
pip install -r requirements.txt
```

---

## 📄 License

For educational / portfolio use.
