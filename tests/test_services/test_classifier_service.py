import unittest
from unittest import mock
from unittest.mock import call
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
        from sklearn.neural_network import MLPClassifier
        params = {
            'model': MLPClassifier,
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


    def test_evaluate(self):
        X = np.array([[1,2,3,4]]*10)
        Y = np.array([True, True, True, False, False, False, False, False, False, False])

        class Model:
            def predict_proba(self, X):
                return np.array([[0.98, 0.02]]*5+[[0.02, 0.98]]*5)

        model = Model()
        metric = cs.evaluate(model, X, Y)
        self.assertAlmostEqual(metric, 0.14285, places=4)
