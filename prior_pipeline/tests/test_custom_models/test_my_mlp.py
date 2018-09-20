import unittest


from custom_models.my_mlp import FunnelMLPClassifier as fmlp

class TestFunnelMLPClassifier(unittest.TestCase):

    def test_set_layers(self):
        layers = fmlp.set_layers(
            input_layer=200,
            num_hidden_layer=2,
            output_layer=100
        )
        self.assertEqual(layers, (200, 166, 133, 100))

        layers = fmlp.set_layers(
            input_layer=100.0,
            num_hidden_layer=2.0,
            output_layer=200.0
        )
        self.assertEqual(layers, (100, 133, 166, 200))
