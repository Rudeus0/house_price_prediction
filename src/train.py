import pandas as pd
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

def load_data() -> pd.DataFrame:
    data_frame = fetch_california_housing(data_farme=(True))
    housing_df = data_frame.frame
    return housing_df


def transfrom_features(housing_df: pd.DataFrame) -> pd.DataFrame:
    housing_dfc = housing_df.copy()
    housing_dfc["AveRooms"] = np.log1p(housing_dfc["AveRooms"])
    housing_dfc["AveBedrms"] = np.log1p(housing_dfc["AveBedrms"])
    housing_dfc["Population"] = np.log1p(housing_dfc["Population"])
    housing_dfc["AveOccup"] = np.log1p(housing_dfc["AveOccup"])
    housing_dfc["MedInc"] = np.log1p(housing_dfc["MedInc"])
    return housing_dfc



