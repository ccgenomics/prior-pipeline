import os
import pandas as pd
import hyperopt as hpo
import numpy as np
from sklearn import metrics


class ClassifierService:

    dflt =  (('model', 'test'),
            ('data_path', 'path_to_data'),
            ('targets_path', 'path_to_targets'))

    cv_fold = 5
    train_set = 0.8

    @classmethod
    def objective(cls, args):
        print(args)
        metrics = []
        for seed in range(1, cls.cv_fold+1):
            model = cls.interpreter(args[0])
            data_path = args[1]
            targets_path = args[2]
            X_train, Y_train, X_test, Y_test = cls.load_data(
                data_path=data_path,
                targets_path=targets_path,
                train_set = cls.train_set,
                random_state=seed
            )
            model.fit(X_train, Y_train)
            metric = cls.evaluate(model, X_test, Y_test)
            metrics.append(metric)
        metrics_mean = np.array(metrics).mean()
        return 1 - metrics_mean


    @classmethod
    def fit(cls, X, Y, space=dflt):
        best = hpo.fmin(
            cls.objective,
            space,
            algo=hpo.tpe.suggest,
            max_evals=100
        )

    @classmethod
    def interpreter(cls, model):
        instance = model['name'](**model['params'] if 'params' in model else {})
        return instance

    @classmethod
    def evaluate(cls, model, X, Y):
        probs = model.predict_proba(X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(Y, probs)
        print(fpr, tpr, thresholds)
        return metrics.auc(fpr, tpr, reorder=False)

    @classmethod
    def load_data(cls, data_path, targets_path, train_set=train_set, random_state=1 ):
        data = pd.read_csv(data_path)
        targets = pd.read_csv(targets_path).iloc[:,-1]
        # print(targets.iloc[:,-1])
        full = pd.concat([data, targets], axis=1, join='inner')
        train_size = int(len(full)*train_set)
        shuffled = full.sample(frac=1, random_state=random_state)
        X_train = shuffled.iloc[:train_size,:-1].as_matrix()
        Y_train = shuffled.iloc[:train_size, -1].as_matrix()
        X_test = shuffled.iloc[train_size:, :-1].as_matrix()
        Y_test = shuffled.iloc[train_size:, -1].as_matrix()
        return X_train, Y_train, X_test, Y_test