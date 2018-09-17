import os
import pandas as pd
import hyperopt as hpo
from sklearn import metrics

class ClassifierService:

    dflt = ('model', 'test')

    @classmethod
    def objective(cls, arg):
        print('done')
        return 1

    @classmethod
    def fit(cls, X, Y, space=dflt):
        best = hpo.fmin(cls.objective, space, algo=hpo.tpe.suggest, max_evals=100)

    @classmethod
    def interpreter(cls, params):
        model = params['model'](**params['params'])
        return model

    @classmethod
    def evaluate(cls, model, X, Y):
        probs = model.predict_proba(X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(Y, probs)
        return metrics.auc(fpr, tpr, reorder=False)
