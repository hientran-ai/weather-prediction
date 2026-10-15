import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils.pipeline import build_training_frame, chronological_split


def train_on_web(df_raw):
    """Train LightGBM + Ridge + Baseline từ CSV upload."""
    df = build_training_frame(df_raw)

    feature_cols = [c for c in df.columns if c not in ['datetime', 'target']]

    train, val, test = chronological_split(df)

    X_tr,  y_tr  = train[feature_cols], train['target']
    X_val, y_val = val[feature_cols],   val['target']
    X_te,  y_te  = test[feature_cols],  test['target']

    # ── Ridge ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_te)

    best_alpha, best_rmse, best_ridge = None, float('inf'), None
    for alpha in [0.01, 0.1, 1, 10, 100, 1000]:
        m = Ridge(alpha=alpha)
        m.fit(X_tr_s, y_tr)
        rmse = np.sqrt(mean_squared_error(y_val, m.predict(X_val_s)))
        if rmse < best_rmse:
            best_rmse, best_alpha, best_ridge = rmse, alpha, m

    ridge_pred    = best_ridge.predict(X_te_s)
    ridge_val_pred = best_ridge.predict(X_val_s)
    ridge_val_metrics = {
        'R2': round(r2_score(y_val, ridge_val_pred), 4),
        'MAE': round(mean_absolute_error(y_val, ridge_val_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_val, ridge_val_pred)), 3),
        'name': 'Ridge - Validation'
    }
    ridge_metrics = {
        'R2':   round(r2_score(y_te, ridge_pred), 4),
        'MAE':  round(mean_absolute_error(y_te, ridge_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_te, ridge_pred)), 3),
        'name': 'Ridge - Test'
    }

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1
    )
    lgbm_model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    lgbm_pred    = lgbm_model.predict(X_te)
    lgbm_val_pred = lgbm_model.predict(X_val)
    lgbm_val_metrics = {
        'R2': round(r2_score(y_val, lgbm_val_pred), 4),
        'MAE': round(mean_absolute_error(y_val, lgbm_val_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_val, lgbm_val_pred)), 3),
        'name': 'LightGBM - Validation'
    }
    lgbm_metrics = {
        'R2':   round(r2_score(y_te, lgbm_pred), 4),
        'MAE':  round(mean_absolute_error(y_te, lgbm_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_te, lgbm_pred)), 3),
        'name': 'LightGBM - Test'
    }

    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline_pred = test['tempmax'].values
    baseline_val_pred = val['tempmax'].values
    baseline_val_metrics = {
        'R2': round(r2_score(y_val, baseline_val_pred), 4),
        'MAE': round(mean_absolute_error(y_val, baseline_val_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_val, baseline_val_pred)), 3),
        'name': 'Baseline - Validation'
    }
    baseline_metrics = {
        'R2':   round(r2_score(y_te, baseline_pred), 4),
        'MAE':  round(mean_absolute_error(y_te, baseline_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_te, baseline_pred)), 3),
        'name': 'Baseline (tempmax hôm nay)'
    }

    split_info = {
        'train_rows': len(train), 'validation_rows': len(val), 'test_rows': len(test),
        'train_end': train['datetime'].max(),
        'validation_end': val['datetime'].max(),
        'test_end': test['datetime'].max(),
    }
    lgbm_bundle = {
        'model': lgbm_model, 'scaler': None, 'feature_cols': feature_cols,
        'metrics': lgbm_metrics, 'validation_metrics': lgbm_val_metrics,
        'model_type': 'lightgbm', 'split_info': split_info,
    }
    ridge_bundle = {
        'model': best_ridge, 'scaler': scaler, 'feature_cols': feature_cols,
        'metrics': ridge_metrics, 'validation_metrics': ridge_val_metrics,
        'model_type': 'ridge', 'best_alpha': best_alpha, 'split_info': split_info,
    }
    baseline_bundle = {
        'metrics': baseline_metrics, 'validation_metrics': baseline_val_metrics,
        'split_info': split_info,
    }
    return lgbm_bundle, ridge_bundle, baseline_bundle, feature_cols
