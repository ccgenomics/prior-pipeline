import unittest
from unittest import mock
from unittest.mock import call
from sklearn.neural_network import MLPClassifier
import numpy as np
import hyperopt

from services.classifier_service import ClassifierService as cs

class TestClassifierService(unittest.TestCase):
    @mock.patch('hyperopt.fmin')
    def test_fit(self, mock_fmin):
        X = np.genfromtxt('./tests/data/A.csv', delimiter=',')
        Y = np.genfromtxt('./tests/data/targets.csv', delimiter=',')
        cs.fit(X, Y)
        mock_fmin.assert_called_once()

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
