from Layer import Layer
import numpy as np

class NeuralNetwork:
    def __init__(self, layers_config, learning_rate=0.01):
        self.layers = []
        for i in range(1, len(layers_config)):
            self.layers.append(Layer(layers_config[i], layers_config[i-1]))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        for layer in self.layers:
            inputs = np.array(layer.forward(inputs))
        return inputs

    def calculate_error(self, expected_output, actual_output):
        return 0.5 * np.sum((expected_output - actual_output) ** 2)

    def backward(self, inputs, expected_output):
        # Calculate gradients and update weights
        deltas = []
        layer_inputs = [np.array(inputs)]  # Inputs for each layer
        
        for layer in self.layers:
            inputs = np.array(layer.forward(inputs))
            layer_inputs.append(inputs)
        
        # Output layer error
        error = expected_output - layer_inputs[-1]
        delta = error * layer_inputs[-1] * (1 - layer_inputs[-1])  # Derivative of sigmoid
        deltas.append(delta)
        
        # Backpropagate errors
        for i in reversed(range(len(self.layers) - 1)):
            delta = deltas[-1].dot(np.array([neuron.weights for neuron in self.layers[i + 1].neurons])) * layer_inputs[i + 1] * (1 - layer_inputs[i + 1])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights
        for i in range(len(self.layers)):
            layer_input = np.atleast_2d(layer_inputs[i])
            delta = np.atleast_2d(deltas[i])
            for j, neuron in enumerate(self.layers[i].neurons):
                neuron.weights += self.learning_rate * layer_input.T.dot(delta[:, j]).flatten()

    def train(self, training_inputs, training_outputs, epochs):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_output in zip(training_inputs, training_outputs):
                actual_output = self.forward(inputs)
                total_error += self.calculate_error(expected_output, actual_output)
                self.backward(inputs, expected_output)
            errors.append(total_error)
            print(f"Epoch {epoch+1}, Error: {total_error}")
        return errors
