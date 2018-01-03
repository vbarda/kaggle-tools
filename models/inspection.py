import logging

from funcy import zipdict
import pandas as pd
from sklearn import clone, metrics
from sklearn.base import is_regressor, is_classifier
from sklearn.pipeline import Pipeline

from models.cross_validation import TrainTestSplitter

logger = logging.getLogger(__name__)

ALLOWED_PROBLEM_TYPES = ('binary_classification', 'regression')
DEFAULT_CLASSIFIER_SCORERS = (metrics.f1_score,
                              metrics.accuracy_score,
                              metrics.roc_auc_score)
DEFAULT_REGRESSOR_SCORERS = (metrics.r2_score,
                             metrics.mean_squared_error,
                             metrics.mean_absolute_error)


def make_default_metrics_dict(scorers):
    ''''''
    return {scorer.__name__: scorer for scorer in scorers}


def get_coefs(feature_names, model):
    '''Make a dict with feature names and coefficients'''
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return zipdict(feature_names, model.coef_)


class ModelInspector(TrainTestSplitter):

    def __init__(self, df, problem_type, estimators_dict, metrics_dict,
                 feature_names, target_name, full_train=False, **cv_kwargs):
        '''Split a dataset into training and test samples, fit multiple
            estimators and compare metrics
        Args:
            df: (pd.DataFrame) source df with features and target
            problem_type: (str) {'binary_classification', 'regression'}
            estimators_dict: (dict) of instantiated sklearn estimators/pipelines
                e.g. {'linear_regression': LinearRegression(),
                      'lasso': LassoCV()}
            feature_names: (list) of feature names
            target_name: (str) name of the column that contains target values
            full_train: (bool) whether to train the estimator on all data,
                default is False (trains only on self.X_train, self.y_train)
            **cv_kwargs: kwargs passed to TrainTestSplitter
        '''
        super(ModelInspector, self).__init__(df, feature_names, target_name)
        super(ModelInspector, self).split(**cv_kwargs)
        self.estimators_dict = estimators_dict
        self.problem_type = problem_type
        self._validate()
        self.metrics_dict = metrics_dict or self.get_default_metrics_dict()
        self.fitted_estimators_dict = {}
        self.full_train = full_train

    def get_default_metrics_dict(self):
        '''Get dictionary of scoring functions that are appropriate for a given
        problem type, e.g. {'r2_score': r2_score}
        '''
        scorers = {
            'regression': DEFAULT_REGRESSOR_SCORERS,
            'binary_classification': DEFAULT_CLASSIFIER_SCORERS
        }[self.problem_type]
        return make_default_metrics_dict(scorers)

    def _validate(self):
        '''Validate the estimators to make sure the correspond to the problem'''
        if self.problem_type not in ALLOWED_PROBLEM_TYPES:
            raise ValueError('Expected problem type to be one of "{}", got '
                             '"{}" instead'.format(ALLOWED_PROBLEM_TYPES,
                                                   self.problem_type))
        estimator_checking_func = {
            'regression': is_regressor,
            'binary_classification': is_classifier
        }[self.problem_type]
        estimators = self.estimators_dict.values()
        if not all(map(estimator_checking_func, estimators)):
            raise ValueError('All estimators should be of "{}" problem type'
                             .format(self.problem_type))

    def fit(self, full_train=None):
        '''fit each estimator in the self.estimators_dict'''
        if full_train is not None:
            self.full_train = full_train
        if self.full_train:
            X, y = self.X, self.y
        else:
            X, y = self.X_train, self.y_train
        for estimator_name, estimator in self.estimators_dict.iteritems():
            logger.info('Fitting estimator "{}"'.format(estimator_name))
            _estimator = clone(estimator)
            _estimator.fit(X, y)
            self.fitted_estimators_dict[estimator_name] = _estimator
        return self

    @staticmethod
    def calculate_metrics(fitted_estimators_dict, metrics_dict, X, y):
        '''produce a dataframe with metrics for each estimator'''
        metrics = {}
        if not fitted_estimators_dict:
            raise AssertionError('Cannot calculate metrics without fitting '
                                 'estimators first')
        for estimator_name, estimator in fitted_estimators_dict.iteritems():
            estimator_metrics = {}
            y_pred = estimator.predict(X)
            for metric_name, scorer in metrics_dict.iteritems():
                estimator_metrics[metric_name] = scorer(y, y_pred)
            metrics[estimator_name] = estimator_metrics
        return pd.DataFrame(metrics)

    @property
    def train_metrics(self):
        return self.calculate_metrics(self.fitted_estimators_dict,
                                      self.metrics_dict,
                                      self.X_train,
                                      self.y_train)

    @property
    def test_metrics(self):
        return self.calculate_metrics(self.fitted_estimators_dict,
                                      self.metrics_dict,
                                      self.X_test,
                                      self.y_test)

    @property
    def full_metrics(self):
        if not self.full_train:
            raise ValueError('Estimator has to be be trained on all data '
                             'to show full_metrics')
        return self.calculate_metrics(self.fitted_estimators_dict,
                                      self.metrics_dict,
                                      self.X,
                                      self.y)
