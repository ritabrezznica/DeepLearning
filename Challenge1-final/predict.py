#!/usr/bin/env python3
"""
Inference script for credit card default prediction.
Loads the trained CreditDefaultMLP and scaler, applies preprocessing
to a new CSV, and outputs a CSV of predictions.

Usage: python predict.py <input_csv> <output_csv>
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load

INPUT_DIM = 34
HIDDEN_DIM = 32
DROPOUT = 0.2

NOMINAL = ["SEX", "EDUCATION", "MARRIAGE"]

NUM_COLS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "AVG_PAY_DELAY", "MAX_PAY_DELAY", "NUM_LATE", "UTIL_RATIO", "PAY_RATIO",
]

FEATURE_COLS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "AVG_PAY_DELAY", "MAX_PAY_DELAY", "NUM_LATE", "UTIL_RATIO", "PAY_RATIO",
    "SEX_1", "SEX_2",
    "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4",
    "MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3",
]


class CreditDefaultMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, hidden_depth=2, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(16, (input_dim + 2) // 2)
        layers = []
        in_f = input_dim
        for _ in range(hidden_depth):
            layers.extend([
                nn.Linear(in_f, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_f = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def preprocess(df, scaler):
    df = df.copy()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "TARGET"})
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    pay_hist = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["AVG_PAY_DELAY"] = df[pay_hist].mean(axis=1)
    df["MAX_PAY_DELAY"] = df[pay_hist].max(axis=1)
    df["NUM_LATE"] = (df[pay_hist] >= 1).sum(axis=1)
    df["UTIL_RATIO"] = df["BILL_AMT1"] / (df["LIMIT_BAL"].abs() + 1e-8)
    df["PAY_RATIO"] = df["PAY_AMT1"] / (df["BILL_AMT1"].abs() + 1e-8)

    dummies = pd.get_dummies(df[NOMINAL], columns=NOMINAL, drop_first=False)
    X_num = df[NUM_COLS].astype(np.float32)
    X_enc = pd.concat([X_num.reset_index(drop=True),
                       dummies.reset_index(drop=True)], axis=1)

    X_enc = X_enc.reindex(columns=FEATURE_COLS, fill_value=0)
    X_scaled = scaler.transform(X_enc)
    return X_scaled.astype(np.float32)


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    scaler = load("scaler.joblib")

    model = CreditDefaultMLP(INPUT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path, header=1)

    X = preprocess(df, scaler)

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        _, preds = torch.max(logits, 1)

    pd.DataFrame({"prediction": preds.numpy()}).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
