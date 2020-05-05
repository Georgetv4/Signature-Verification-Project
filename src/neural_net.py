"""
This file describes Neural Net working
"""
from src.config import DEBUG
from src.neuron import Neuron


class NeuralNet:
    def __init__(self, *n_layers):
        self.layers = [[]] * len(n_layers)
        n_count = n_layers[0]
        self.layers[0] = [Neuron(n_count)] * n_count

        for i in range(1, len(n_layers)):
            n_count = n_layers[i - 1]
            self.layers[i] = [Neuron(n_count)] * n_layers[i]

    def ff(self, inputs):  # Forward propagation
        next_input = None
        current_input = inputs

        for i in range(len(self.layers)):
            if DEBUG:
                print(current_input)

            next_input = [0] * len(self.layers[i])
            for j in range(len(self.layers[i])):
                next_input[j] = self.layers[i][j].ff(current_input)

            current_input = next_input.copy()

        return next_input

    def save_model(self):
        for layer in self.layers:
            neuron_list = []
            for neuron in layer:
                neuron_list.append(str(neuron))

            print(";".join(neuron_list))

    def load_model(self):
        pass
