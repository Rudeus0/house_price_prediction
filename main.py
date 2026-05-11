from src.train import load_data, transform_features, split_and_scale, train_model
from src.evaluate import evaluate_models

if __name__ == "__main__":
    # 1. Load
    housing_df = load_data()
    
    # 2. Transform
    housing_dfc = transform_features(housing_df)
    
    # 3. Split and scale
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(
        housing_df, housing_dfc
    )
    
    # 4. Train
    model_lr, model_rf = train_model(X_train_scaled, y_train)
    
    # 5. Evaluate
    results = evaluate_models(model_lr, model_rf, X_test_scaled, y_test)