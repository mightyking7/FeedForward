from random import random
from math import exp


class FeedForward:
    """
    Fully connected feed forward network
    used to classify MNIST dataset
    :author Isaac Buitrago
    """
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.network = list()

        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)

        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)


    def fit(self, dataset, lr, n_epochs, n_outputs):

        # train model on dataset
        for epoch in range(n_epochs):
            for row in dataset:
                outputs = self._forward(row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                self._backwards(expected)
                self._update_weights(row, lr)

    def predict(self, row):
        outputs = self._forward(row)
        return outputs.index(max(outputs))

    def accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def _activate(self, weights, inputs):
        """
        Calculate neuron activation for an input
        """
        active = weights[-1]
        for i in range(len(weights) - 1):
            active += weights[i] * inputs[i]
        return active


    def _forward(self, row):
        """
        Forward propagate a row of inputs through the network
        """
        inputs = row
        for layer in self.network:
            new_inputs = []

            for neuron in layer:
                activation = self._activate(neuron['weights'], inputs)
                neuron['output'] = self._transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def _transfer(self, activation):
        """
        Implements sigmoid to determine if a nueron
        is fired or not
        """
        return 1.0 / (1.0 + exp(-activation))

    def _backwards(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self._transfer(neuron['output'])

    def _update_weights(self, row, lr):

        for i in range(len(self.network)):
            inputs = row[:-1]

            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]

            for neuron in self.network[i]:

                for j in range(len(inputs)):
                    neuron['weights'][j] += lr * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += lr * neuron['delta']