"""
Main script for training and evaluating DBN model.
"""

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from DBN import DBN
import warnings
import os

warnings.filterwarnings('ignore')


def load_data(data_path, feature_cols, target_col):
    """Load dataset from CSV and return features, target, and dataframe."""
    df = pd.read_csv(data_path)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.reshape(-1, 1).astype(np.float32)
    return X, y, df


def train_and_evaluate(X_train, y_train, X_val, y_val, scaler_y, device,
                      hidden_units, batch_size, epoch_pre, epoch_ft):
    """Train DBN and return metrics plus inverse-transformed predictions."""
    dbn = DBN(hidden_units, X_train.shape[1], 1, device=device)
    dbn.pretrain(X_train, epoch=epoch_pre, batch_size=batch_size)
    dbn.finetune(X_train, y_train, epoch_ft, batch_size,
                 torch.nn.MSELoss(), torch.optim.Adam(dbn.parameters()))

    y_pred_scaled = dbn.predict(X_val, batch_size).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }, y_true, y_pred


def main():
    # ========== Configuration (modify as needed) ==========
    cfg = {
        'data_path': 'data/dataset.csv',          # path to input CSV
        'feature_cols': ['f1', 'f2', 'f3', 'f4', 'f5',
                         'f6', 'f7', 'f8', 'f9', 'f10'],  # update with actual names
        'target_col': 'target',                    # update with actual target name
        'hidden_units': [256, 128, 64],
        'batch_size': 200,
        'epoch_pretrain': 200,
        'epoch_finetune': 200,
        'n_splits': 5,
        'test_size': 0.2,
        'output_dir': 'results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    # ======================================================

    os.makedirs(cfg['output_dir'], exist_ok=True)
    print(f"Using device: {cfg['device']}")

    # Load and standardize data
    X, y, _ = load_data(cfg['data_path'], cfg['feature_cols'], cfg['target_col'])
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).reshape(-1, 1)

    # ---- K-fold cross validation ----
    kf = KFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
    all_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_scaled)):
        metrics, _, _ = train_and_evaluate(
            X_scaled[tr_idx], y_scaled[tr_idx],
            X_scaled[va_idx], y_scaled[va_idx],
            scaler_y, cfg['device'], cfg['hidden_units'],
            cfg['batch_size'], cfg['epoch_pretrain'], cfg['epoch_finetune']
        )
        metrics['fold'] = fold + 1
        all_metrics.append(metrics)
        print(f"Fold {fold+1} | MSE: {metrics['mse']:.4f} | "
              f"MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")

    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_r2  = np.mean([m['r2']  for m in all_metrics])
    print(f"\nAverage CV | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f} | R²: {avg_r2:.4f}")

    # ---- Train final model on full data ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_scaled, test_size=cfg['test_size'], random_state=42
    )

    final_model = DBN(cfg['hidden_units'], X_scaled.shape[1], 1, device=cfg['device'])
    final_model.pretrain(X_tr, epoch=cfg['epoch_pretrain'], batch_size=cfg['batch_size'])
    final_model.finetune(X_tr, y_tr, cfg['epoch_finetune'], cfg['batch_size'],
                         torch.nn.MSELoss(), torch.optim.Adam(final_model.parameters()))

    # Save model
    model_path = os.path.join(cfg['output_dir'], 'model.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'scaler_X_mean': scaler_X.mean_,
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_mean': scaler_y.mean_,
        'scaler_y_scale': scaler_y.scale_,
    }, model_path)
    print(f"\nModel saved to: {model_path}")

    # Test set evaluation
    y_te_pred = final_model.predict(X_te, cfg['batch_size']).reshape(-1, 1)
    y_te_pred_inv = scaler_y.inverse_transform(y_te_pred).flatten()
    y_te_inv = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()

    print(f"Test MSE: {mean_squared_error(y_te_inv, y_te_pred_inv):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_te_inv, y_te_pred_inv):.4f}")
    print(f"Test R²:  {r2_score(y_te_inv, y_te_pred_inv):.4f}")


if __name__ == '__main__':
    main()