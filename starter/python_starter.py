from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submission" / "starter_predictions.csv"

FEATURES = [
    "ApplicantAge", "EmploymentYears", "AnnualIncome", "RequestedAmount",
    "AnnualPayment", "HouseholdSize", "VendorScoreAtlas", "VendorScoreBeacon",
    "VendorScoreCedar", "CreditInquiries12M", "LoanToIncomeRatio",
    "PaymentToIncomeRatio", "PriorAccountCount", "ActiveAccountCount",
    "OutstandingBalance", "OverdueAccountCount", "CreditHistoryYears",
    "PriorApplicationCount", "PriorApprovalRate", "DebtToIncomeRatio",
]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

def main():
    train = pd.read_csv(DATA / "application_train.csv")
    evaluation = pd.read_csv(DATA / "application_evaluation.csv")
    med = train[FEATURES].median(numeric_only=True)
    mean = train[FEATURES].fillna(med).mean()
    std = train[FEATURES].fillna(med).std().replace(0, 1)

    def matrix(df):
        x = ((df[FEATURES].fillna(med) - mean) / std).to_numpy(float)
        return np.column_stack([np.ones(len(x)), x])

    x = matrix(train)
    y = train["DefaultFlag"].to_numpy(float)
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    rate = 0.08
    for step in range(700):
        p = sigmoid(x @ beta)
        gradient = x.T @ (p - y) / len(y)
        gradient[1:] += 0.002 * beta[1:]
        beta -= rate * gradient
        if step in (250, 500):
            rate *= 0.55

    prediction = sigmoid(matrix(evaluation) @ beta)
    pd.DataFrame({
        "CustomerID": evaluation["CustomerID"],
        "PredictedDefaultProbability": prediction,
    }).to_csv(OUT, index=False)
    print(f"Wrote {len(prediction):,} predictions to {OUT}")

if __name__ == "__main__":
    main()
