import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_predictions(df: pd.DataFrame, horizon: int) -> np.ndarray:
    days_of_week = df.index.dayofweek.astype('category')
    X = pd.get_dummies(days_of_week, drop_first=True)
    y = df.y
    model = LinearRegression().fit(X, y)

    preds_start_date = df.index[-1] + pd.tseries.frequencies.Day()
    preds_dates = pd.date_range(preds_start_date, freq='D', periods=horizon)
    preds_days_of_week = preds_dates.dayofweek.astype(days_of_week.dtype)
    X_pred = pd.get_dummies(preds_days_of_week, drop_first=True)
    return model.predict(X_pred)
