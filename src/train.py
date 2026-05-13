import pandas as pd
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def load_data() -> pd.DataFrame:
    data_frame = fetch_california_housing(as_frame=(True))
    housing_df = data_frame.frame
    return housing_df


def transform_features(housing_df: pd.DataFrame) -> pd.DataFrame:
    housing_dfc = housing_df.copy()
    housing_dfc["AveRooms"] = np.log1p(housing_dfc["AveRooms"])
    housing_dfc["AveBedrms"] = np.log1p(housing_dfc["AveBedrms"])
    housing_dfc["Population"] = np.log1p(housing_dfc["Population"])
    housing_dfc["AveOccup"] = np.log1p(housing_dfc["AveOccup"])
    housing_dfc["MedInc"] = np.log1p(housing_dfc["MedInc"])
    return housing_dfc

def split_and_scale(housing_df: pd.DataFrame, housing_dfc: pd.DataFrame) -> tuple:
    X = housing_dfc.drop(["MedHouseVal"], axis=1) 
    y = housing_df["MedHouseVal"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler



def train_model(X_train_scaled: np.ndarray, y_train: pd.Series ) -> tuple:
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    return model_lr, model_rf

