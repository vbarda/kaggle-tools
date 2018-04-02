from contextlib import closing
from multiprocessing import Pool, cpu_count

import pandas as pd

from utils.validation import validate_iterable


def parallel_groupby_apply(df, groupby, func, n_cpus=None):
    '''Groupby and apply using multiple cores

    Args:
        df: (pd.DataFrame)
        groupby: (list) of columns to group by
        func: (func) that will be applied to each group
        n_cpus: (int) how many CPUs to use. Defaults to maximum available
    '''
    validate_iterable(groupby)
    g = df.groupby(groupby)
    n_cpus = n_cpus or cpu_count()
    keys = [grp_name for grp_name, _ in g]
    with closing(Pool(n_cpus)) as p:
        processed_dfs = p.map(func, [grp_df for _, grp_df in g])
    concatted = pd.concat(processed_dfs, keys=keys)
    existing_idx_names = list(concatted.index.names)
    idx_names = groupby + existing_idx_names[len(groupby):]
    return concatted.rename_axis(idx_names)


def add_next_periods_padding(df, n_periods=1, freq=None):
    '''Add n extra empty rows for next n_periods

    Args:
        df: (pd.DataFrame) needs to be indexed by date
        n_periods: (int) for how many next periods to add empty rows
        freq: (str) if None, will try to infer from existing datetime index
    '''
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(u'Expected df to have index of DatetimeIndex type, got {} instead'
                        .format(type(df.index)))
    freq = freq or df.index.freq or 'D'
    next_n_periods_index = pd.bdate_range(start=df.index.max(),
                                          periods=n_periods + 1, freq=freq)[-n_periods:]
    if isinstance(df, pd.DataFrame):
        df_to_append = pd.DataFrame(columns=df.columns, index=next_n_periods_index)
    elif isinstance(df, pd.Series):
        df_to_append = pd.Series(index=next_n_periods_index)
    else:
        raise TypeError(u'Expected df to be either pd.DataFrame or pd.Series, got {} instead'.format(type(df)))
    return df.append(df_to_append)
