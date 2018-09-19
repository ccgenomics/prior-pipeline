import unittest
from unittest import mock
from unittest.mock import call
from sklearn.neural_network import MLPClassifier
from hyperopt.mongoexp import MongoTrials
import numpy as np
import hyperopt
import pandas as pd

from services.classifier_service import ClassifierService as cs
from services.download_service import DownloadService as ds

class TestClassifierService(unittest.TestCase):
    @mock.patch('hyperopt.fmin')
    def test_fit(self, mock_fmin):
        cs.fit()
        mock_fmin.assert_called_once()

    @mock.patch('hyperopt.fmin')
    @mock.patch('hyperopt.mongoexp.MongoTrials')
    def test_fit_mongo(self,mock_MongoTrials, mock_fmin):
        mongo = {
            'url': 'mongo://localhost:1234/foo_db/jobs',
            'exp': 'my_experience'
        }
        mock_MongoTrials.return_value = 'random_hash_4b9u4y5h3r9u'
        cs.fit(mongo=mongo)
        mock_MongoTrials.assert_called_once_with('mongo://localhost:1234/foo_db/jobs', exp_key='my_experience')
        args, kwargs = mock_fmin.call_args_list[0]
        self.assertEqual(kwargs['trials'],  'random_hash_4b9u4y5h3r9u')

    def test_interpreter(self):
        params = {
            'name': MLPClassifier,
            'params': {
                'activation': 'relu',
                'hidden_layer_sizes': (4,3,2)
            }
        }
        expected_model = MLPClassifier(
            activation='relu',
            hidden_layer_sizes=(4,3,2)
        )

        model = cs.interpreter(params)
        self.assertEqual(str(expected_model), str(model))

    def test_interpreter_no_params(self):

        params = {
            'name': MLPClassifier
        }
        expected_model = MLPClassifier()

        model = cs.interpreter(params)
        self.assertEqual(str(expected_model), str(model))


    def test_evaluate(self):
        X = np.array([[1,2,3,4]]*5 + [[2,4,6,8]]*5 )
        Y = np.array([True, True, True, False, False, False, False, False, False, False])

        class Model:
            def predict_proba(self, X):
                return np.array([[0.98, 0.02]]*5+[[0.02, 0.98]]*5)

        model = Model()
        metric = cs.evaluate(model, X, Y)
        self.assertAlmostEqual(metric, 0.14285, places=4)

    @mock.patch('services.classifier_service.ClassifierService.load_data')
    def test_objective(self, mock_load_data):
        params = ({
            'name': MLPClassifier,
            'params': {
                'activation': 'relu',
                'hidden_layer_sizes': (4,3,2),
                'random_state':1
                }
            },
            './tests/data/A.csv',
            './tests/data/targets.csv')

        mock_load_data.return_value = (
            [[1,2,3]]*5 + [[3,4,5]]*5 , [[1]]*4+[[0]]*6
        )*2
        cv_metric = cs.objective(params)
        calls= [
            call(   data_path='./tests/data/A.csv',
                    targets_path='./tests/data/targets.csv',
                    train_set = 0.8,
                    random_state=i
            ) for i in range(1, 6)
            ]
        mock_load_data.assert_has_calls(calls)
        self.assertEqual(cv_metric, .5)


    def test_load_data(self):
        X_train, Y_train, X_test, Y_test = cs.load_data(
            data_path='./tests/data/A.csv',
            targets_path='./tests/data/targets.csv',
            train_set = 0.5,
            random_state=1
        )
        self.assertEqual(str(X_train), str(np.genfromtxt('./tests/data/exp_train_A.csv', delimiter=',', dtype=int)))
        self.assertEqual(str(Y_train), str(np.genfromtxt('./tests/data/exp_train_targets.csv', delimiter=',', dtype=bool)))
        self.assertEqual(str(X_test), str(np.genfromtxt('./tests/data/exp_test_A.csv', delimiter=',', dtype=int)))
        self.assertEqual(str(Y_test), str(np.genfromtxt('./tests/data/exp_test_targets.csv', delimiter=',', dtype=bool)))

    @mock.patch('services.classifier_service.ClassifierService.get_data')
    def test_load_data_calls_get_data(self, mock_get_data):
        data_path = './tests/data/A.csv'
        targets_path = './tests/data/targets.csv'
        mock_get_data.return_value = pd.read_csv(data_path, index_col=0)
        X_train, Y_train, X_test, Y_test = cs.load_data(
            data_path=data_path,
            targets_path=targets_path,
            train_set = 0.5,
            random_state=1
        )
        mock_get_data.assert_called_once_with(data_path)

    def test_get_data(self):
        data = cs.get_data('./tests/data/A.csv')
        exp_data = pd.read_csv('./tests/data/A.csv', index_col=0)
        self.assertEquals(str(data), str(exp_data))

    @mock.patch('services.download_service.DownloadService.download_file')
    @mock.patch('pandas.read_csv')
    def test_get_data_not_found_locally(self, mock_read_csv, mock_download_file):
        data = cs.get_data('./tests/data/B.csv')
        mock_download_file.assert_called_once_with(
            'ccg-machine-learning',
            'prior-data/B.csv',
            './tests/data/B.csv'
        )
