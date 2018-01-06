from multiprocessing import Pool, cpu_count

import pandas as pd


def parallel_apply(gb, func):
    '''Parallelize groupby'''
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [grp_df for grp_name, grp_df in gb])
    return pd.concat(ret_list)