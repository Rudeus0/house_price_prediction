import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_models(model_lr, model_rf, X_test_scaled, y_test) -> dict:
    
    # Linear Regression
    y_pred_lr = model_lr.predict(X_test_scaled)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    # Random Forest
    y_pred_rf = model_rf.predict(X_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    results = {
        "Linear Regression": {"RMSE": rmse_lr, "R2": r2_lr},
        "Random Forest": {"RMSE": rmse_rf, "R2": r2_rf},
    }

    for model_name, metrics in results.items():
        print(f"{model_name:<25} RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")

    return results