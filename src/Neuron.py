import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0.0

    def activate(self, inputs):
        # Suma ponderada
        self.output = np.dot(inputs, self.weights) + self.bias
        # Aplicar función sigmoide
        self.output = 1 / (1 + np.exp(-self.output))
        return self.output
