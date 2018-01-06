import pandas as pd
import sklearn
from tqdm import tqdm
import xgboost as xgb


def fit_group_models(df, feature_names, target_name, groupby, estimator, use_xgboost=False, xgb_params=None):
    '''Fit separate models for each group in df

    Args:
        df: (pd.DataFrame) that contains features and target
        feature_names: (iterable) of feature names
        target_name: (str)
        groupby: (iter | str) column(s) to groubpy on
        estimator: (sklearn estimator)
        use_xgboost: (bool) if True, DMatrix and xgb.train will be used instead of sklearn syntax
        xgb_params: (dict) if use_xgboost=True, these will be used in xgb.train

    Returns:
        dict of fitted models: {grp_name: fitted_group_estimator, ...}
    '''
    fitted_models = {}
    for grp_name, grp_df in tqdm(df.groupby(groupby)):
        if use_xgboost:
            dtrain = xgb.DMatrix(df[feature_names], label=df[target_name])
            model = xgb.train(xgb_params, dtrain)
        else:
            model = sklearn.clone(estimator).fit(grp_df[feature_names], grp_df[target_name])
        fitted_models[grp_name] = model
    return fitted_models


def predict_with_group_models(df, feature_names, groupby, fitted_group_models, index_cols, use_xgboost=False):
    '''Make predictions using fitted models

    Args:
        df: (pd.DataFrame) that contains features and target
        feature_names: (iterable) of feature names that were used in training
        groupby: (iter | str) column(s) to groupby on
        fitted_group_models: (dict) of fitted models: {grp_name: fitted_group_estimator, ...}
        index_cols: (iter) additional columns from the original df to be added to predictions
        use_xgboost: (bool) if True, DMatrix will be used for making x_df

    Returns:
        pd.DataFrame of predictions
    '''
    grp_preds_dfs = []
    for grp_name, grp_df in tqdm(df.groupby(groupby)):
        x_df = xgb.DMatrix(grp_df[feature_names]) if use_xgboost else grp_df[feature_names]
        if grp_name not in fitted_group_models:
            continue
        grp_preds = fitted_group_models[grp_name].predict(x_df)
        grp_preds_df = grp_df.set_index(index_cols).assign(yhat=grp_preds)[['yhat']]
        grp_preds_dfs.append(grp_preds_df)
    return pd.concat(grp_preds_dfs)