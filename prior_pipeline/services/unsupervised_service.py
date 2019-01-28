import os
from hyperopt import hp, tpe, fmin
import hyperopt as hpo
import random
import numpy as np

class UnsupervisedService():

    def load_csvs(self, csv_directory):
        X = []
        file_names = os.listdir(csv_directory)
        file_names.sort()
        for file_name in file_names:
            if file_name.endswith('.csv'):
                x = self.load_csv(csv_directory+file_name)
                X.append(x)
        return X

    def load_csv(self, csv_path, header=True):
        X = []
        with open(csv_path) as csv_file:
            for line in csv_file:
                if header == True:
                    header=False
                    continue
                values = line.strip('\n').split(',')
                float_values = list(map(float, values))
                X.append(float_values)
        return X

    @classmethod
    def interpreter(self, model_definition):
        cls_name = model_definition['name']
        init_arg = model_definition['params'] if 'params' in model_definition else {}
        return cls_name(**init_arg)


    def load_data(self, data_path=None, seed=42, train_rate=.7):
        X = self.load_csvs(csv_directory=data_path)
        random.seed(seed)
        random.shuffle(X)
        train_size = int(len(X) * train_rate)

        return X[:train_size], X[train_size:]

    @classmethod
    def objective(cls, args):
        model_definition = args[0]
        data_path = args[1]
        cv_iter = 5
        metrics = []
        for i in range(cv_iter):
            model = cls.interpreter(model_definition)
            X_train, X_test = cls().load_data(data_path=data_path, seed=i)
            model.fit(X_train)
            metrics.append(model.evaluate(X_test))

        return np.asarray(metrics).mean()

    @classmethod
    def fit(cls, space=(), max_evals=100):
        best = hpo.fmin(
            cls.objective,
            space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=None
        )
        return best
