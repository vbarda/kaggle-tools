import pandas as pd
from sklearn import clone
from sklearn.model_selection import train_test_split, KFold

DEFAULT_TEST_SIZE = .3
DEFAULT_RANDOM_STATE = 1234


class TrainTestSplitter(object):
    def __init__(self, df, feature_names, target_name):
        '''Split the data into train and test sets, and store the splits in
        sklearn-friendly format
        Args:
            df: (pd.DataFrame) source df with features and target
            feature_names: (list) default None - all columns but target are
                considered features
            target_name: (str) name of the column that contains target values
        '''
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df should only be pd.DataFrame, got "{}" instead'
                            .format(type(df)))
        self.df = df
        if feature_names is None:
            feature_names = self.df.columns.drop(target_name)
        self.feature_names = feature_names
        self.target_name = target_name

        # subset df into X and y
        self.X = self.df[feature_names]
        self.y = self.df[target_name]

    def split(self, test_size=DEFAULT_TEST_SIZE,
              random_state=DEFAULT_RANDOM_STATE, shuffle=True):
        '''split into training and test samples
        Args:
            test_size: (float) share of the data that will be used as a test set
            random_state: (int) passed to
                sklearn.model_selection.train_test_split
            shuffle: (bool) passed to sklearn.model_selection.train_test_split
        '''
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = train_test_split(self.X, self.y,
                                         test_size=test_size,
                                         random_state=random_state,
                                         shuffle=shuffle)
        return self


def get_kfold_cv_scores(n_splits, shuffle, estimator, X, y, agg_func):
    '''performs K-fold cross validation on a specified number of subsamples
    Args:
        n_splits: (int) number of subsamples used for cross validation
        shuffle: (bool) passed to sklearn.model_selection.KFold
        estimator: (string) name of the sklearn estimator
        X: (pd.DataFrame) part of the source df with features
        y: (pd.Series) part of the source df with target values
        agg_func: (func) function to aggregate scores from each fold
    '''
    score_list = []
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=shuffle).split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        _estimator = clone(estimator)
        _estimator.fit(X_train, y_train)
        score = _estimator.score(X_test, y_test)
        score_list.append(score)
    return agg_func(score_list)