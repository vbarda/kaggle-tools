from attrdict import AttrDict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import pandas as pd
from tqdm import tqdm

from models.cross_validation import CrossValidation
from tools.helpers import binarize

DEFAULT_METRICS = f1_score, accuracy_score, roc_auc_score

class ModelInspector(CrossValidation):

    def __init__(self, df, models, feature_names=None, metrics=DEFAULT_METRICS, full_train=False, 
                 **cv_kwargs):
        '''
        Split a dataset into training and test samples, fit multiple models and compare metrics
        Args:
            df: (pd.DataFrame) source df with features and target
            models: (dict) of named scikit models / pipelines that have fit and predict method
            feature_names: (list) default None - all columns but target are considered features
            metrics: (list) of scikit metrics to score the predictions
            full_train: (bool) whether to train the model on all data, default is False
            **cv_kwargs: kwargs passed to CrossValidation
        '''
        super(ModelInspector, self).__init__(df, feature_names, **cv_kwargs)
        self.models = AttrDict(models)
        self.full_train = full_train

    def fit(self, full_train=False):
        '''fit each model in the self.models dictionary'''
        self.full_train = full_train or self.full_train
        X, y = self.X_train, self.y_train
        if self.full_train:
            X, y = self.X, self.y
        for model_name, model in tqdm(self.models.items()):
            model.fit(X, y)
            setattr(self, model_name, model)
        return self
    
    def _metrics(self, X, y):
        '''produce a dataframe with metrics for each model'''
        return pd.DataFrame({model_name: model_metrics(model, X, y)
                            for model_name, model in self.models.items()})
    
    @property
    def train_metrics(self):
        return self._metrics(self.X_train, self.y_train)

    @property
    def test_metrics(self):
        return self._metrics(self.X_test, self.y_test)

    @property
    def full_metrics(self):
        if not self.full_train:
            raise ValueError('the model has to be be trained on all data to show full_metrics')
        return self._metrics(self.X, self.y)


def model_metrics(model, X, y, metrics=DEFAULT_METRICS):
    '''Return a dictionary of metrics for a model'''
    preds = binarize(model.predict(X))
    return {metric.__name__: metric(y, preds) for metric in metrics}

