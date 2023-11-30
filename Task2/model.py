from mlp_algorithm import *
from preprocessing import *
def train_test_model(function_type, num_layers, num_neurons, eta, m, isBias):
    X_train, Y_train, X_test, Y_test = Preprocessing(filepath='..//data//Dry_Bean_Dataset.xlsx')
    input_size = 5
    num_neurons = [int(num) for num in num_neurons.split(",")]
    output_size = 3
    num_hidden_layer = num_neurons
    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(input_size, output_size, X_train, num_hidden_layer, Y_train, function_type, eta, m, isBias)
    # train network
    weights, biases, train_accuracy = mlp.train()
    print('Weights = ', weights)
    print('Biases = ', biases)
    print("Train Accuracy = ", train_accuracy, "%")
    test_accuracy = mlp.testing_accuracy(function_type, X_test, Y_test)*100
    print("Test Accuracy = ", test_accuracy, "%")
    print("Confusion Matrix List = ", mlp.calculate_confusion_matrix(function_type, X_test, Y_test))