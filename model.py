import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def dow_month_interaction(dows, months):
    interaction = [100*month + dow for month, dow in zip(months, dows)]
    return np.asarray(interaction)[:, None]


def get_predictions(df: pd.DataFrame, horizon: int) -> np.ndarray:
    months = df.index.month.values
    days_of_week = df.index.dayofweek.values
    X = dow_month_interaction(days_of_week, months)

    dow_categories = list(range(7))
    month_categories = [x+1 for x in range(12)]
    categories = []
    for month in month_categories:
        for dow in dow_categories:
            categories.append(100*month + dow)
    categories = sorted(categories)

    model = make_pipeline(OneHotEncoder(categories=[categories], dtype='uint8'),
                          LinearRegression())
    y = df.y.values
    model.fit(X, y)

    preds_start_date = df.index[-1] + pd.tseries.frequencies.Day()
    preds_dates = pd.date_range(preds_start_date, freq='D', periods=horizon)
    preds_dows = preds_dates.dayofweek
    preds_months = preds_dates.month
    X_preds = dow_month_interaction(preds_dows, preds_months)
    return model.predict(X_preds)
