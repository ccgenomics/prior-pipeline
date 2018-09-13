import unittest
from unittest import mock
from unittest.mock import call

from services.projection_service import ProjectionService as ps

class TestProjectionService(unittest.TestCase):
    @mock.patch("pandas.DataFrame.to_csv")
    def test_project(self, mock_to_csv):
        path = './tests/data/A.csv'
        ps.project(path, dims=[2,3,4], output='./data/output')
        calls= [
            call('./data/output/A_TSNE_2D.csv'),
            call('./data/output/A_TSNE_3D.csv'),
            call('./data/output/A_TSNE_4D.csv')
            ]
        mock_to_csv.assert_has_calls(calls)
