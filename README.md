### kaggle-tools

Tools for kaggle competitions


*Cross Validation*:

```
from models.cross_validation import TrainTestSplitter
splitter = TrainTestSplitter(df).split(test_size=.2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(splitter.X_train, splitter.y_train)
log_reg.predict(splitter.X_test)
```

*Model Inspector*:

```
from models.inspection import ModelInspection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score

clf_dict = {
    'log_reg': LogisticRegression(),
    'knn_5': KNeighborsClassifier(n_neighbors=5),
    'knn_10': KNeighborsClassifier(n_neighbors=10)
}

metrics_dict = {
   'f1_score': f1_score,
   'roc_auc': roc_auc_score
}

mi = ModelInspector(df, 'binary_classification', clf_dict, metrics_dict,
                    None, 'target', False, test_size=.2).fit()
mi.test_metrics.loc['roc_auc_score'].sort_values(ascending=False)
```
