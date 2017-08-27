import pandas as pd
from sklearn.model_selection import train_test_split

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