#!/usr/bin/env python3
"""
Inference script for credit card default prediction.
Usage: python predict.py <input_csv> <output_csv>
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load


class CreditDefaultNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        # Architecture: [32, 16] with SELU, AlphaDropout(0.1), no BatchNorm
        layers.append(nn.Linear(33, 32))
        layers.append(nn.SELU())
        layers.append(nn.AlphaDropout(0.1))
        layers.append(nn.Linear(32, 16))
        layers.append(nn.SELU())
        layers.append(nn.AlphaDropout(0.1))
        layers.append(nn.Linear(16, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def preprocess(df, scaler):
    df = df.copy()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    if 'default payment next month' in df.columns:
        df = df.drop(columns=['default payment next month'])

    # Clean categoricals
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
    df['SEX'] = df['SEX'].map({1: -1, 2: 1})

    # One-hot encode
    df = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'], prefix=['EDU', 'MAR'])
    ohe_cols = [c for c in df.columns if c.startswith('EDU_') or c.startswith('MAR_')]
    df[ohe_cols] = df[ohe_cols].replace({0: -1})

    # Make sure all expected categorical columns exist
    for c in ['SEX', 'EDU_1', 'EDU_2', 'EDU_3', 'EDU_4', 'MAR_1', 'MAR_2', 'MAR_3']:
        if c not in df.columns:
            df[c] = -1

    # Numerical columns (same order as training)
    num_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    cat_cols = ['SEX', 'EDU_1', 'EDU_2', 'EDU_3', 'EDU_4', 'MAR_1', 'MAR_2', 'MAR_3']

    # Scale numerical features
    df[num_cols] = scaler.transform(df[num_cols])

    # Feature engineering
    pay_cols = [c for c in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] if c in df.columns]
    df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
    df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
    df['NUM_LATE'] = (df[pay_cols] > 0).sum(axis=1).astype(float)
    df['UTIL_RATIO'] = df['BILL_AMT1'] / (df['LIMIT_BAL'].abs() + 1e-8)
    df['PAY_RATIO'] = df['PAY_AMT1'] / (df['BILL_AMT1'].abs() + 1e-8)

    eng_cols = ['AVG_PAY_DELAY', 'MAX_PAY_DELAY', 'NUM_LATE', 'UTIL_RATIO', 'PAY_RATIO']

    # Assemble in correct column order (must match training)
    all_cols = num_cols + cat_cols + eng_cols
    return df[all_cols].values.astype(np.float32)


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    scaler = load('scaler.pkl')

    model = CreditDefaultNet()
    model.load_state_dict(torch.load('best.pth', map_location='cpu'))
    model.eval()

    df = pd.read_csv(input_path) if input_path.endswith('.csv') else pd.read_excel(input_path, header=1)
    X = preprocess(df, scaler)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy().ravel()

    preds = (probs >= 0.5).astype(int)
    pd.DataFrame({'prediction': preds}).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path} ({len(preds)} rows)")


if __name__ == '__main__':
    main()
