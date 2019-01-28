import unittest
from unittest import mock
from unittest.mock import call, MagicMock
from hyperopt import hp, tpe

from services.unsupervised_service import UnsupervisedService as us

class TestUnsupervisedService(unittest.TestCase):

    def test_load_csvs(self):
        csv_directory = './tests/data/multiple_csv/'
        X = us().load_csvs(csv_directory)
        print(X)

        self.assertEqual(
            X,
            [
                [[4.0,2,2,9],[9.3,4.3,1.1,1]],
                [[3.0,1,1,8],[2.3,3.3,0.1,0],[2.3,3.3,0.1,0]]
            ]
        )

    def test_load_csv(self):
        csv_path = './tests/data/multiple_csv/csv_1.csv'
        X = us().load_csv(csv_path)
        self.assertEqual(
            X,
            [[4.0,2,2,9],[9.3,4.3,1.1,1]]
        )

    @mock.patch('hyperopt.fmin')
    def test_fit(self, mock_fmin):
        space = (
            hp.choice('model',
            [
                {
                    'name': 'MyUnsupervisedModel',
                    'params':{
                        'alpha': hp.quniform('alpha', 0, 10, 1)
                    }
                }
            ]),
            hp.choice('data_path', ['./tests/data/mutliple_csv'])
            )
        us().fit(space=space, max_evals=100)
        mock_fmin.assert_called_once_with(
            us.objective,
            space,
            max_evals=100,
            algo=tpe.suggest,
            trials=None
        )


    @mock.patch('services.unsupervised_service.UnsupervisedService.interpreter')
    @mock.patch('services.unsupervised_service.UnsupervisedService.load_data')
    def test_objective(self, mock_load_data, mock_interpreter):
        model = MagicMock()
        mock_interpreter.return_value = model
        mock_load_data.return_value = [[1]],[[2]]
        args = (
                    {
                    'name': 'MyUnsupervisedModel',
                    'params': { 'alpha': 0.1 }
                    },
                'path_to_data'
                )

        us.objective(args)

        calls = [
            call(data_path='path_to_data', seed=0),
            call(data_path='path_to_data', seed=1),
            call(data_path='path_to_data', seed=2),
            call(data_path='path_to_data', seed=3),
            call(data_path='path_to_data', seed=4)
        ]
        mock_load_data.assert_has_calls(calls)

        calls = [
            call.fit([[1]]),
            call.evaluate([[2]]),
            call.fit([[1]]),
            call.evaluate([[2]]),
            call.fit([[1]]),
            call.evaluate([[2]]),
            call.fit([[1]]),
            call.evaluate([[2]]),
            call.fit([[1]]),
            call.evaluate([[2]])
        ]

        model.assert_has_calls(calls)

    @mock.patch('services.unsupervised_service.UnsupervisedService.load_csvs')
    def test_load_data(self, mock_load_csvs):
        X = [  [ [1,2],[1,4] ], [ [1,2],[1,4],[3,7] ], [ [7, 5] ] ]
        mock_load_csvs.return_value = X
        X_train, X_test = us().load_data(
            data_path='./path_to_data/',
            seed=1,
            train_rate=.8)

        mock_load_csvs.assert_called_once_with(csv_directory='./path_to_data/')
        self.assertEqual(X_train, [[ [1,2],[1,4],[3,7] ], [ [7, 5] ]])
        self.assertEqual(X_test, [[ [1,2],[1,4] ]])
