import numpy as np
import random


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # first layer is the input layer, so no bias and weights defined for that

    def feedForward(self, a):
        """forward propagation of a neural network"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        "stochastic gradient descent."
        test_samples = 0
        if test_data is not None:
            test_samples = len(test_data)
        total_samples = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batch_list = [training_data[k: k + mini_batch_size]
                               for k in range(0, total_samples, mini_batch_size)]
            #             mini_batch_list = []
            #             for k in range(0,total_samples, mini_batch_size):
            #                 mini_batch_list.append(training_data[ k : k + mini_batch_size])
            for mini_batch in mini_batch_list:
                self.updateMiniBatch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}/{2}".format(i, self.evaluate(test_data), test_samples))
            print("Epoch {0} completed".format(i))

    def updateMiniBatch(self, mini_batch, eta):

        nebla_b = [np.zeros(b.shape) for b in self.biases]
        nebla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nebla_b, delta_nebla_w = self.backPropagation(x, y)
            nebla_b = [nb + dnb for nb, dnb in zip(nebla_b, delta_nebla_b)]
            nebla_w = [nw + dnw for nw, dnw in zip(nebla_w, delta_nebla_w)]
        self.weights = [w - (eta / len(mini_batch) * nw)
                        for w, nw in zip(self.weights, nebla_w)]
        self.biases = [b - (eta / len(mini_batch) * nb)
                       for b, nb in zip(self.biases, nebla_b)]

    def evaluate(self, test_data):
        """evaluate results, run the forward pass and compare the results with actual labels"""
        test_results = [(np.argmax(self.feedForward(x)), y) for x, y in test_data]
        result = sum(int(x == y) for x, y in test_results)
        return result

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def backPropagation(self, x, y):
        """ return gradient of cost function C_x in form of (nabla_b, nabla_w)"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation_list = [x]
        z_list = []
        current_activation = x
        # feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, current_activation) + b  # np.dot here actually do the multiplication
            z_list.append(z)
            current_activation = sigmoid(z)
            activation_list.append(current_activation)

        # Backward pass
        delta = self.cost_derivative(activation_list[-1], y) * sigmoid_prime(z_list[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activation_list[-2].transpose())

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z_list[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activation_list[-l - 1].transpose())
        return (nabla_b, nabla_w)

def sigmoid(z):
    return (1.0/(1.0+ np.exp(-z)))

def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))

