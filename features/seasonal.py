import pandas as pd

from utils.validation import validate_iterable

def add_period_dummies(df, periods, datetime_col, drop_first=True):
    '''Add period dummies'''
    validate_iterable(periods)
    period_df = df.copy()
    for period in periods:
        period_df[period] = getattr(period_df[datetime_col].dt, period)
    return pd.get_dummies(period_df, columns=periods, drop_first=drop_first)