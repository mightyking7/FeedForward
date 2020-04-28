import numpy as np
from math import exp
from random import random, seed
from sklearn.metrics import log_loss


class FeedForward:
    """
    Fully connected feed forward network
    used to classify MNIST dataset
    :author Isaac Buitrago
    """
    def __init__(self, n_inputs, n_hidden, n_outputs):

        # seed random number generator
        seed(1)

        self.n_outputs = n_outputs
        self.network = list()

        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)

        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)


    def fit(self, x, y, lr, n_epochs):
        """
        Trains network with backpropagation using stochastic gradient descent
        :param x: images to train on
        :param y: target vector with labels
        :param lr: learning rate
        :param n_epochs: training epochs
        """
        # train model on dataset
        for epoch in range(n_epochs):
            err = 0
            for img, target in zip(x, y):
                outputs = self._forward(img)
                # one hot encode label
                target = np.array([0 if i != target else 1 for i in range(self.n_outputs)]).reshape((-1, 1))
                err += log_loss(target, outputs)
                self._backwards(target)
                self._update_weights(img, lr)
            print(f"> Epoch = {epoch}, lrate = {lr}, error = {err:.3f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Used to classify the input
        :param x: input to classify
        :return: classification of each image as an ndarray
        """
        classes = np.zeros((len(x),), dtype = int)

        # classify input batch
        for i in range(len(x)):
            outputs = self._forward(x[i, :])
            classes[i] = outputs.argmax()

        return classes

    def _activate(self, weights, inputs):
        """
        Calculate neuron activation for an input
        """
        # bias
        active = weights[-1]

        for i in range(len(weights) - 1):
            active += weights[i] * inputs[i]

        return active


    def _forward(self, x : np.ndarray) -> np.ndarray:
        """
        Forward propagate a row of inputs through the network
        and returns a probability distribution over the classes.
        :img: vectorized image
        :returns ndarray of probabilities for ten classes
        """
        inputs = x
        for layer in self.network:
            new_inputs = []

            for neuron in layer:
                activation = self._activate(neuron['weights'], inputs)
                neuron['output'] = self._transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs

        inputs = np.array(inputs)

        # probability distribution over classes
        classes = self._softmax(inputs).reshape((-1, 1))

        return classes

    def _transfer(self, activation):
        """
        Transfers neuron activation using Sigmoid function
        """
        return 1.0 / (1.0 + exp(-activation))

    def _transfer_derivative(self, output):
        """
        Calculate derivative of neuron output
        :param output: neuron output
        :return:
        """
        return output * (1.0 - output)

    def _backwards(self, target):
        """
        backward propagates the error and stores error
        between target and output in each neuron
        :param target:
        """
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()

            # hidden layer
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                # output layer
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(target[j, :] - neuron['output'])

            # store error in neurons
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self._transfer_derivative(neuron['output'])

    def _update_weights(self, x, lr):

        for i in range(len(self.network)):

            inputs = x

            # input is output from previous layer
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]

            # update weights in current layer
            for neuron in self.network[i]:

                # update neuron weights
                for j in range(len(inputs)):
                    neuron['weights'][j] += lr * neuron['delta'] * inputs[j]

                # update bias
                neuron['weights'][-1] += lr * neuron['delta']

    def _softmax(self, outputs: np.ndarray) -> np.ndarray:
        """
        Applies softmax over network outputs
        :param outputs: output layer
        :return: ndarray of probability distribution over classes
        """
        exps = np.exp(outputs)
        return exps / exps.sum()