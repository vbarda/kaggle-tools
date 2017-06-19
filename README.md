### kaggle-tools

Tools for kaggle competitions


*Cross Validation*:

```
from models.cross_validation import CrossValidation
cv = CrossValidation(df)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(cv.X_train, cv.y_train)
log_reg.predict(cv.X_test)
```

*Model Inspector*:

```
from models.inspection import ModelInspection
from sklearn.linear_model import LogisticRegression, LinearRegression

models = {
	'lin_reg': LinearRegression(),
    'log_reg': LogisticRegression(),
    'lasso': LassoCV()
}

mi = ModelInspector(df, models, test_size=.2).fit()
mi.test_metrics.loc['roc_auc_score'].sort_values(ascending=False)
```
