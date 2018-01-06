from multiprocessing import Pool, cpu_count

import pandas as pd


def parallel_apply(gb, func):
    '''Parallelize groupby'''
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [grp_df for grp_name, grp_df in gb])
    return pd.concat(ret_list)


def add_next_periods_padding(df, n_periods=1, freq=None):
    '''Add n extra empty rows for next n_periods

    Args:
        df: (pd.DataFrame) needs to be indexed by date
        n_periods: (int) for how many next periods to add empty rows
        freq: (str) if None, will try to infer from existing datetime index
    '''
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(u'Expected df to have index of DatetimeIndex type, got {} instead'.format(type(df.index)))
    freq = freq or df.index.freq or 'D'
    next_n_periods_index = pd.bdate_range(start=df.index.max(), periods=n_periods + 1, freq=freq)[-n_periods:]
    if isinstance(df, pd.DataFrame):
        df_to_append = pd.DataFrame(columns=df.columns, index=next_n_periods_index)
    elif isinstance(df, pd.Series):
        df_to_append = pd.Series(index=next_n_periods_index)
    else:
        raise TypeError(u'Expected df to be either pd.DataFrame or pd.Series, got {} instead'.format(type(df)))
    return df.append(df_to_append)
