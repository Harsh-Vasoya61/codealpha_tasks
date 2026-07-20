"""
Realistic Synthetic Credit Dataset Generator
==============================================
Generates a larger (20,000 row), more realistic credit dataset than the
original version:
  - Correlated features (income grows with age/employment, credit limit
    scales with income, etc.)
  - A more realistic ~15-18% default rate (real-world portfolios are NOT
    50/50 - the original toy version was unrealistically balanced)
  - Non-linear + interaction effects in the risk model (so tree-based
    models like Random Forest have a genuine edge over Logistic Regression,
    like in real credit data)
  - Missing values in a couple of columns (income, employment length) to
    mimic real bureau data quality issues
  - More categorical variables: education, marital status, employment type
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_realistic_credit_data(n_samples=20000):
    # --- Demographics ---
    age = np.clip(np.random.normal(40, 12, n_samples), 21, 75).round().astype(int)

    education = np.random.choice(
        ["high_school", "bachelors", "masters", "phd"],
        n_samples, p=[0.35, 0.40, 0.20, 0.05]
    )
    edu_income_multiplier = pd.Series(education).map(
        {"high_school": 0.8, "bachelors": 1.0, "masters": 1.3, "phd": 1.5}
    ).values

    marital_status = np.random.choice(
        ["single", "married", "divorced", "widowed"],
        n_samples, p=[0.35, 0.45, 0.15, 0.05]
    )

    employment_type = np.random.choice(
        ["salaried", "self_employed", "unemployed", "retired"],
        n_samples, p=[0.65, 0.18, 0.07, 0.10]
    )

    # Employment length correlated with age, but capped and zero for unemployed
    base_emp_length = np.clip((age - 21) * np.random.uniform(0.15, 0.6, n_samples), 0, 40)
    employment_length = np.where(employment_type == "unemployed", 0, base_emp_length).round(1)
    employment_length = np.where(employment_type == "retired",
                                  np.clip(employment_length, 5, 40), employment_length)

    # --- Income: correlated with age (rises then plateaus), education, employment type ---
    age_income_factor = 1 + np.clip((age - 21) / 40, 0, 1) * 0.9
    employment_income_factor = pd.Series(employment_type).map(
        {"salaried": 1.0, "self_employed": 1.1, "unemployed": 0.25, "retired": 0.55}
    ).values
    annual_income = (
        28000 * age_income_factor * edu_income_multiplier * employment_income_factor
        * np.random.lognormal(mean=0, sigma=0.35, size=n_samples)
    ).round(2)

    # --- Credit history ---
    credit_history_length = np.clip(age - np.random.randint(18, 26, n_samples), 0, None)
    num_open_accounts = np.clip(
        np.random.poisson(lam=2 + credit_history_length / 10, size=n_samples), 0, 25
    )

    # Credit limit scales with income and history length
    credit_limit = (
        annual_income * np.random.uniform(0.15, 0.5, n_samples)
        * (1 + credit_history_length / 50)
    ).round(2)

    # Utilization: right-skewed, higher for financially stressed profiles
    credit_utilization = np.clip(np.random.beta(2, 4, n_samples), 0, 1).round(3)

    # Existing debt driven by utilization * limit + extra installment debt
    existing_debt = (
        credit_limit * credit_utilization
        + np.random.lognormal(mean=8.0, sigma=1.0, size=n_samples)
    ).round(2)

    # Payment history: late payments more likely with high utilization / low income
    financial_stress = (credit_utilization * 2 + (existing_debt / (annual_income + 1))).clip(0, 5)
    num_late_payments_12m = np.random.poisson(lam=0.15 + 0.6 * financial_stress, size=n_samples)
    num_late_payments_12m = np.clip(num_late_payments_12m, 0, 12)

    num_credit_inquiries_6m = np.clip(
        np.random.poisson(lam=0.5 + 0.3 * financial_stress, size=n_samples), 0, 15
    )

    # --- Loan request ---
    loan_amount_requested = (
        annual_income * np.random.uniform(0.05, 0.4, n_samples)
    ).round(2)

    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN"], n_samples, p=[0.38, 0.42, 0.20]
    )
    purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "car", "medical", "other"],
        n_samples, p=[0.30, 0.22, 0.14, 0.14, 0.10, 0.10]
    )

    # --- Latent default risk (non-linear + interactions) ---
    debt_to_income = existing_debt / (annual_income + 1)
    loan_to_income = loan_amount_requested / (annual_income + 1)

    risk_score = (
        2.2 * debt_to_income
        + 2.8 * credit_utilization
        + 0.45 * num_late_payments_12m
        + 0.12 * num_credit_inquiries_6m
        + 1.0 * loan_to_income
        - 0.035 * credit_history_length
        - 0.000006 * annual_income
        - 0.02 * employment_length
        # interaction: high utilization AND many late payments compounds risk
        + 0.6 * (credit_utilization > 0.7) * (num_late_payments_12m > 1)
        # unemployment is a strong independent risk driver
        + 1.3 * (employment_type == "unemployed")
        + np.random.normal(0, 0.6, n_samples)
    )

    # Calibrate to a realistic ~16% default rate
    prob_default = 1 / (1 + np.exp(-(risk_score - np.quantile(risk_score, 0.84))))
    default = np.random.binomial(1, prob_default)

    df = pd.DataFrame({
        "age": age,
        "education": education,
        "marital_status": marital_status,
        "employment_type": employment_type,
        "employment_length_years": employment_length,
        "annual_income": annual_income,
        "credit_history_length": credit_history_length,
        "num_open_accounts": num_open_accounts,
        "credit_limit": credit_limit,
        "credit_utilization": credit_utilization,
        "existing_debt": existing_debt,
        "num_late_payments_12m": num_late_payments_12m,
        "num_credit_inquiries_6m": num_credit_inquiries_6m,
        "loan_amount_requested": loan_amount_requested,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "default": default,
    })

    # --- Inject realistic missingness (real bureau data is never fully clean) ---
    income_missing_idx = df.sample(frac=0.04, random_state=RANDOM_STATE).index
    df.loc[income_missing_idx, "annual_income"] = np.nan

    emp_missing_idx = df.sample(frac=0.02, random_state=RANDOM_STATE + 1).index
    df.loc[emp_missing_idx, "employment_length_years"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_realistic_credit_data(n_samples=20000)
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    df.to_csv("credit_data_realistic.csv", index=False)
    print("\nSaved to credit_data_realistic.csv")
