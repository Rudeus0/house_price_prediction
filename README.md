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

## Bugs Encountered and Fixed

### Bug 1 — Wrong sklearn parameter
```python
# WRONG
fetch_california_housing(data_farme=(True))

# CORRECT
fetch_california_housing(as_frame=True)
```
Typo in parameter name caused silent failure — dataset loaded without frame structure.

---

### Bug 2 — Log transform applied to target variable
```python
# WRONG — MedHouseVal got transformed
housing_dfc = housing_df.copy()
y = housing_dfc['MedHouseVal']   # log-transformed target

# CORRECT — target stays in original scale
y = housing_df['MedHouseVal']    # original housing_df, not housing_dfc
```
This caused R² of -0.52 — model appeared completely broken. Root cause: predictions were in log scale but evaluation expected original scale. Fix: always take `y` from the untransformed dataframe.

---

### Bug 3 — Train/test split before log transforms
```python
# WRONG — split on original data, transforms applied after
X, y = housing_df.drop(...), housing_df['MedHouseVal']
X_train, X_test, ... = train_test_split(X, y)
housing_dfc["MedInc"] = np.log1p(...)  # too late

# CORRECT — transform first, split after
housing_dfc["MedInc"] = np.log1p(...)  # transform first
X, y = housing_dfc.drop(...), housing_df['MedHouseVal']
X_train, X_test, ... = train_test_split(X, y)  # split after
```
Caused model to train on unlogged features but test on different distribution.

---

### Bug 4 — `scaler.fit()` instead of `scaler.transform()` on test set
```python
# WRONG — fits a new scaler on test data (data leakage)
X_test_scaled = scaler.fit(X_test)

# CORRECT — only transform, never fit on test data
X_test_scaled = scaler.transform(X_test)
```
Fitting on test data leaks information from the future into the model. Always `fit_transform` on train, `transform` only on test.

---

### Bug 5 — All code in one file
Initially `train.py`, `evaluate.py` and `main.py` logic were all in one file. Caused circular import errors and `exited with code=0` with no output.

Fix: separated into three files with single responsibilities.

---

### Bug 6 — `np.sqrt()` missing from RMSE
```python
# WRONG — returns MSE not RMSE
rmse = mean_squared_error(y_test, y_pred)

# CORRECT
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

---

