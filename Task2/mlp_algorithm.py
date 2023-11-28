import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class MLP(object):
    """A Multilayer Perceptron class.
    """
    def __init__(self, num_inputs, num_outputs, inputs, hidden_layers, targets, function_type, learning_rate, epochs, isBias):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = inputs
        self.hidden_layers = len(hidden_layers)
        self.targets = targets
        self.function_type = function_type
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.isBias = isBias
        layers = []
        if (hidden_layers == [0]):
            layers = [self.num_inputs] + [self.num_outputs]
            self.hidden_layers = 0
        else:
            layers = [self.num_inputs] + hidden_layers + [self.num_outputs]
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(np.random.rand(layers[i + 1], layers[i]))
            biases.append(np.random.rand(layers[i + 1], 1))
        if(self.isBias == 0):
            for i in range(len(biases)):
                biases[i] = np.zeros(biases[i].shape)
        self.weights = weights
        self.biases = biases
    def forward_propagate(self, inputs, activation_function):
        net = []
        net.append(inputs.reshape(len(inputs), 1))
        for i in range(self.hidden_layers + 1):
            if i == 0:
                net.append(activation_function(
                    np.dot(self.weights[i], inputs.reshape(len(inputs), 1)) + self.biases[i]))
            else:
                net.append(activation_function(np.dot(self.weights[i], net[i]) + self.biases[i]))
        return net
    def back_propagate(self, target, activation_function_derivative, net):
        sigmas = []
        expected_output = np.zeros((self.num_outputs, 1))
        for i in range(self.num_outputs):
            if i == target:
                expected_output[i] = 1
            else:
                expected_output[i] = 0
        sigmas.append((expected_output - net[-1]) * activation_function_derivative(net[-1]))
        for i in range(self.hidden_layers):
            sigmas.append(np.dot(self.weights[-i - 1].T, sigmas[i]) * activation_function_derivative(net[-i - 2]))
        return sigmas
    def train(self):
        activation_function, activation_function_derivative = self.get_activation_function(self.function_type)
        for i in range(self.epochs):
            for j in range(len(self.inputs)):
                net = self.forward_propagate(self.inputs[j], activation_function)
                sigmas = self.back_propagate(self.targets[j], activation_function_derivative, net)
                self.weights, self.biases = self.update_weights(sigmas, net)
        return self.weights, self.biases, self.training_accuracy(activation_function)*100
    def update_weights(self, sigmas, layer_output):
        for i in range(self.hidden_layers + 1):
            self.weights[i] += self.learning_rate * np.dot(sigmas[self.hidden_layers - i], layer_output[i].T)
            if self.isBias == 1:
                self.biases[i] += self.learning_rate * sigmas[self.hidden_layers - i]
        return self.weights, self.biases
    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
    def tanh(self, x):
        return np.tanh(x)
    def tanh_derivative(self, x):
        return 1 - x ** 2
    def get_activation_function(self, function_type):
        if function_type == "sigmoid":
            return self.sigmoid, self.sigmoid_derivative
        else:
            return self.tanh, self.tanh_derivative
    def training_accuracy(self, activation_function):
        predictions = []
        predictions = self.accuracy(self.inputs, activation_function)
        accuracy = accuracy_score(self.targets, predictions)
        return accuracy
    def testing_accuracy(self, activation_function, inputs, outputs):
        if activation_function != self.sigmoid and activation_function != self.tanh:
            activation_function, activation_function_derivative = self.get_activation_function(activation_function)
        predictions = []
        predictions = self.accuracy(inputs, activation_function)
        accuracy = accuracy_score(outputs, predictions)
        return accuracy
    def accuracy(self, inputs, activation_function):
        pred = []
        for i in range(len(inputs)):
            layer_output = self.forward_propagate(inputs[i], activation_function)
            predicted_class = np.argmax(layer_output[-1])
            pred.append(predicted_class)
        return pred

    def build_confusion_matrix(self, confusion_matrix_list, train_accuracy, test_accuracy):
        fig = plt.figure(figsize=(7, 7), facecolor='lightskyblue',
                         layout='constrained')
        ax = fig.add_subplot()
        cax = ax.matshow(confusion_matrix_list, cmap='tab20c')
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        for i in range(self.num_outputs):
            for j in range(self.num_outputs):
                ax.text(j, i, f"{confusion_matrix_list[i, j]:.0f}", ha='center', va='center')
        tick_marks = np.arange(self.num_outputs)
        plt.xticks(tick_marks, ['BOMBAY', 'CALI', 'SIRA'])
        plt.yticks(tick_marks, ['BOMBAY', 'CALI', 'SIRA'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.figtext(0.5, 0.05, f"Train Accuracy: {train_accuracy:.2%},     Test Accuracy: {test_accuracy:.2%}", fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5), ha="center")
        plt.show()

    def calculate_confusion_matrix(self, activation_function, test_samples, test_labels):
        if activation_function != self.sigmoid and activation_function != self.tanh:
            activation_function, activation_function_derivative = self.get_activation_function(activation_function)
        confusion_matrix_list = np.zeros((3, 3))
        for i in range(len(test_samples)):
            layer_output = self.forward_propagate(test_samples[i], activation_function)
            confusion_matrix_list[test_labels[i]][np.argmax(layer_output[-1])] += 1
        self.build_confusion_matrix(confusion_matrix_list, self.training_accuracy(activation_function), self.testing_accuracy(activation_function, test_samples, test_labels))
        return confusion_matrix_list

