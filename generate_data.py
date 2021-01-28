import numpy as np
import pandas as pd

def generate_data(n_weeks=200, seed=0):
    rs = np.random.RandomState(seed)
    n_samples = 7 * n_weeks
    seasonal = np.arange(7).tolist() * n_weeks
    df = pd.DataFrame({
        'ds': pd.date_range('2010-01-01', freq='D', periods=n_samples),
        'y': rs.rand(n_samples) + np.hstack(seasonal)
        })
    month_rands = dict(enumerate(rs.random(12) * 0.5 + 0.5))
    df['month'] = df.ds.dt.month - 1
    df['month_val'] = df.month.map(month_rands)
    df['y'] = df['y'] * df['month_val']
    return df.set_index('ds')[['y']]
