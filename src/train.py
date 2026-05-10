import pandas as pd
import numpy as np 
from sklearn.datasets import fetch_california_housing


def load_data() -> pd.DataFrame:
    data_frame = fetch_california_housing(data_farme=(True))
    housing_df = data_frame.frame
    return housing_df

