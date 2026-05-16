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


## Dataset

- **Source:** `sklearn.datasets.fetch_california_housing`
- **Size:** 20,640 rows × 8 features
- **Target:** `MedHouseVal` (Median house value in $100,000s)

| Feature | Description |
| :--- | :--- |
| **MedInc** | Median income in block group |
| **HouseAge** | Median house age |
| **AveRooms** | Average rooms per household |
| **AveBedrms** | Average bedrooms per household |
| **Population** | Block group population |
| **AveOccup** | Average occupants per household |
| **Latitude** | Block group latitude |
| **Longitude** | Block group longitude |

---

## Feature Engineering & Transformations

To improve model stability—especially for Linear Regression—element-wise log transformations ($log(x + 1)$) were applied to handle highly skewed features and compress extreme outliers (e.g., maximum values in `Population` and `AveOccup`).

Log1p transforms applied to 5 skewed features:

```python
housing_dfc["AveRooms"]    = np.log1p(housing_dfc["AveRooms"])
housing_dfc["AveBedrms"]   = np.log1p(housing_dfc["AveBedrms"])
housing_dfc["Population"]  = np.log1p(housing_dfc["Population"])
housing_dfc["AveOccup"]    = np.log1p(housing_dfc["AveOccup"])
housing_dfc["MedInc"]      = np.log1p(housing_dfc["MedInc"])
```

Target `MedHouseVal` kept in original scale — transforming the target breaks R² evaluation.

---

## Key Insight — Feature Importance

`MedInc` (median income) is the strongest predictor of house price by a large margin. Location (Latitude/Longitude) is second. This confirms the real estate principle: income level of a neighbourhood drives property values more than physical house characteristics.

---

## How to Run

```bash
git clone https://github.com/Rudeus0/house_price_prediction.git
cd house_price_prediction

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
python main.py
```

---

