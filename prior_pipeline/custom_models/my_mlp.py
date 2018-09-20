from sklearn.neural_network import MLPClassifier
import math

class RectangleMLPClassifier(MLPClassifier):
    def __init__(self, hidden_layer_width=2, hidden_layer_depth=2, **kwargs):
        hidden_layer_sizes = tuple([ int(hidden_layer_width)] * int(hidden_layer_depth))
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, **kwargs)

class FunnelMLPClassifier(MLPClassifier):
    def __init__(self,input_layer=100, num_hidden_layer=2, output_layer=2,  **kwargs):
        layers = self.set_layers(
            input_layer=100,
            num_hidden_layer=2,
            output_layer=2)
        super().__init__(hidden_layer_sizes=layers, **kwargs)


    @classmethod
    def set_layers(cls, input_layer=200,num_hidden_layer=2,output_layer=100):
        input_layer = int(input_layer)
        num_hidden_layer = int(num_hidden_layer)
        output_layer = int(output_layer)
        layers = []
        num_layer = num_hidden_layer + 2
        scale = (input_layer - output_layer) / (num_layer-1)
        for i in range(num_layer):
            layers.append(input_layer-math.ceil(scale*i))
        return tuple(layers)
