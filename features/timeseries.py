import pandas as pd

from utils.validation import validate_iterable


def make_ar_features(df, colname, lags=[1]):
    ''''''
    # validate inputs
    validate_iterable(lags)
    # validate datetime index / datetime column
    if isinstance(df.index, pd.DatetimeIndex):
        ar_df = df.sort_index()
    elif 'date' in df:
        ar_df = df.set_index('date').sort_index()
    else:
        raise AssertionError(u'Couldn\'t find datetime index / "date" column')
    ar_feature_names = []
    for lag in lags:
        feature_name = '{}___{}'.format(colname, lag)
        ar_feature_names.append(feature_name)
        ar_df[feature_name] = df[colname].shift(lag)
    return ar_df[ar_feature_names]