from sklearn.model_selection import ParameterGrid
import xgboost as xgb

from models.cross_validation import CrossValidation

# TODO: actually needs XGBCV

class XGBSearch(CrossValidation):

    def __init__(self, df, param_grid):
        '''initialize the CrossValidation and DMatrix objects'''
        super(XGBSearch, self).__init__(df, feature_names)
        # use XGB's DMatrix
        self.full_d_matrix = xgb.DMatrix(self.X, self.y)
        self.train_d_matrix = xgb.DMatrix(self.X_train, self.y_train)
        self.test_d_matrix = xgb.DMatrix(self.X_test, self.y_test)
        # initialize param_grid
        self.param_grid = param_grid
        self.pg = ParameterGrid(param_grid=self.param_grid)

    def fit(self, full_train):
	'''loop over each combination of parameters in self.param_grid'''
        self.full_train = full_train or self.full_train
        X, y = self.X_train, self.y_train
        if self.full_train:
            X, y = self.X, self.y
        for params in tqdm(self.pg):
            # TODO need to combine default parameters with the CV parameters at this point
            xgb.train(params=params)
            # setattr(self, model_name, model)
        return self
