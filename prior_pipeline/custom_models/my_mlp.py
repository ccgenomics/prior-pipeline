from sklearn.neural_network import MLPClassifier

class MyMLPClassifier(MLPClassifier):
    def __init__(self, hidden_layer_width=2, hidden_layer_depth=2, **kwargs):
        hidden_layer_sizes = tuple([ int(hidden_layer_width)] * int(hidden_layer_depth))
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
