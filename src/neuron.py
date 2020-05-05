"""
This file describes Neuron working
"""
from src.config import EXP
import random as rm


def sigmoid(x):
    return 1. / (1 + EXP ** (-x))


class Neuron:
    def __init__(self, input_count, bias=0):
        self.weights = [rm.random() for i in range(input_count)]
        self.bias = bias

    def ff(self, inputs):  # Forward propagation
        output = 0
        for i in range(len(inputs)):
            output += inputs[i] * self.weights[i]

        return sigmoid(output + self.bias)

    def __str__(self):
        weights_to_str = list(map(str, self.weights))
        return ",".join(weights_to_str)
