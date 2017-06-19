import pandas as pd
from sklearn.cross_validation import train_test_split

RANDOM_STATE = 5

class CrossValidation(object):

    def __init__(self, df, feature_names=None, target_name='target', test_size=.4,
                 random_state=RANDOM_STATE, **kwargs):
        '''Split the data into training and test samples:
           Args:
               df: (pd.DataFrame) source df with features and target
               feature_names: (list) default None - all columns but target are considered features
               target_name: (str) name of the column that contains target values
               test_size: (float) share of the data wthat will be used as a test sample
               **kwargs: passes to sklearn.cross_validation.train_test_split
        '''
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df should only be pd.DataFrame, {} passed instead'.format(type(df)))
        self.df = df
        if feature_names is None:
            feature_names = df.columns.drop(target_name)
        self.feature_names = feature_names
        self.target_name = target_name

        # subset df into X and y
        self.X = self.df[feature_names]
        self.y = self.df[target_name]
        
        # split into training and test samples
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=test_size,
                                                                                **kwargs)
