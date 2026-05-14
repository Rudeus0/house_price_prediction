# House Price Prediction — California Housing Dataset

End-to-end ML pipeline predicting California house prices using Linear Regression and Random Forest. Built with scikit-learn, pandas, and NumPy following a modular function-based architecture.

---

## Results

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | 0.7136 | 0.6114 |
| Random Forest | 0.5049 | 0.8055 |

Random Forest explains **80% of house price variance** vs 61% for Linear Regression. RMSE of 0.50 means predictions are off by ~$50,000 on average (prices in $100,000 units).

---
## Project Structure

```
house_price_prediction/
├── notebooks/
│   └── analysis.ipynb       # EDA, feature engineering exploration
├── plots/
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── src/
│   ├── train.py             # load, transform, split, scale, train
│   └── evaluate.py          # RMSE and R² evaluation for both models
├── main.py                  # pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Pipeline Flow

```
load_data()           → fetch California Housing from sklearn
transform_features()  → log1p transforms on skewed columns
split_and_scale()     → 80/20 split + StandardScaler
train_model()         → Linear Regression + Random Forest
evaluate_models()     → RMSE + R² for both models
```

---
