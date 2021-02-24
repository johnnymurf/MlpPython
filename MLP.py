import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MLP:

    def __init__(self, num_of_inputs, num_of_hidden, num_of_outputs, learning_rate, is_sigmoid=True):
        self.learning_rate = learning_rate
        self.hidden = np.zeros(shape=(num_of_hidden, 1,))
        self.is_sigmoid = is_sigmoid
        # create random weights between input -> hidden, hidden -> output
        self.weights_ih = np.random.uniform(low=-1.0, high=1.0, size=(num_of_hidden, num_of_inputs))
        self.weights_ho = np.random.uniform(low=-1.0, high=1.0, size=(num_of_outputs, num_of_hidden))

        # create vector of biases for hidden, and output
        self.bias_hidden = np.random.uniform(low=-1.0, high=1.0, size=(num_of_hidden, 1))
        self.bias_output = np.random.uniform(low=-1.0, high=1.0, size=(num_of_outputs, 1))

    def feedforward(self, input):
        # Generate hidden layer
        # multiply and sum the input by the weights
        self.hidden = np.dot(self.weights_ih, input)

        # add the bias
        self.hidden = self.bias_hidden + self.hidden
        # apply activation function to all elements in hidden layer
        if self.is_sigmoid:
            self.hidden = sigmoid(self.hidden)
        else:
            self.hidden = tanh(self.hidden)
        # Generate output layer
        # multiply and sum hidden layer by output weights
        output = np.dot(self.weights_ho, self.hidden)
        # add the biases
        output = np.add(self.bias_output, output)

        # return output
        if self.is_sigmoid:
            return sigmoid(output)
        else:
            return tanh(output)

    def train(self, input, targets):
        # compute error at output layer
        output = self.feedforward(input)
        # print("output shape: ", output.shape)

        output_error = targets - output

        # gradient descent -> might be wrong here
        if self.is_sigmoid:
            gradient = sigmoid(output, True)
        else:
            gradient = tanh(output, True)
        gradient = gradient * output_error
        gradient = gradient * self.learning_rate

        # calculate hidden -> output deltas
        hidden_transpose = np.transpose(self.hidden)
        weight_ho_deltas = gradient * hidden_transpose

        # adjust weights
        self.weights_ho = self.weights_ho + weight_ho_deltas
        self.bias_output = self.bias_output + gradient

        # compute the error for hidden layers
        weights_ho_transpose = np.transpose(self.weights_ho)
        hidden_errors = np.dot(weights_ho_transpose, output_error)

        # compute hidden gradient
        if self.is_sigmoid:
            hidden_gradient = sigmoid(self.hidden, True)
        else:
            hidden_gradient = tanh(self.hidden, True)
        hidden_gradient = hidden_gradient * hidden_errors
        hidden_gradient = hidden_gradient * self.learning_rate

        # calculate input - > hidden deltas
        inputs_transpose = np.transpose(input)
        weight_ih_deltas = hidden_gradient * inputs_transpose

        # adjust weights
        self.weights_ih = self.weights_ih + weight_ih_deltas
        self.bias_hidden = self.bias_hidden + hidden_gradient


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)

    return 1 / (1 + (np.exp(-x)))


def tanh(x, derivative=False):
    if derivative:
        return 1 - (x ** 2)
    return np.tanh(x)


#
# Question 1
#
# create an multilayer perceptron
# params -> number of inputs, number of hidden nodes, number of output nodes, learning rate
mlp_xor = MLP(2, 3, 1, 0.2)
f = open("output.txt", 'r+')
f.truncate(0)
# data           target
# 0, 1              1
# 1, 0              1
# 0, 0              0
# 1, 1              0
xor_data = [np.array([[0], [1]]), np.array([[1], [0]]), np.array([[0], [0]]), np.array([[1], [1]])]
xor_target = [1, 1, 0, 0]

# train mlp # changing to low number while working on other questions Change back to high!
num_of_epochs = 40000

# Question 2
for epoch in range(num_of_epochs):
    # randomising the training
    random_num = random.randint(0, 3)
    mlp_xor.train(xor_data[random_num], xor_target[random_num])
print("XOR testing:", file=f)
print(str(mlp_xor.feedforward(np.array([[0], [1]]))) + ":    1", file=f)
print(str(mlp_xor.feedforward(np.array([[1], [0]]))) + ":    1", file=f)
print(str(mlp_xor.feedforward(np.array([[1], [1]]))) + ":    0", file=f)
print(str(mlp_xor.feedforward(np.array([[0], [0]]))) + ":    0", file=f)


#
# Question 3
#
print("\n", file=f)

mlp_q3 = MLP(4, 5, 1, 0.1, False)

q3_data = np.random.uniform(low=-1, high=1, size=(4, 200))
q3_target = np.zeros(shape=(1, 200))
# # fill target with sin(x1 - x2 + x3 - x4)
for j in range(200):
    q3_target[:, j] = np.sin(q3_data[0, j] - q3_data[1, j] + q3_data[2, j] - q3_data[3, j])

# training change back to 700
num_of_epochs = 500
for i in range(num_of_epochs):
    error = 0
    error2 = 0
    for j in range(150):
        random_num = random.randint(0, 150)
        mlp_q3.train(np.array(q3_data[:, [random_num]]), np.array(q3_target[:, [random_num]]))
        # using mean squared error
        error += abs(np.array(q3_target[:, [random_num]]) - mlp_q3.feedforward(np.array(q3_data[:, [random_num]])))

    if i % 50 == 0 or i == num_of_epochs - 1:
        error = error / 150

        print("Training Error Q3: " + str(error) + "  Epoch:" + str(i), file=f)
# testing Question 3
error = 0
for j in range(150, 200):
    error += abs(np.array(q3_target[:, [j]]) - mlp_q3.feedforward(np.array(q3_data[:, [j]])))

print("Testing error Q3:    " + str(error / 50), file=f)

#
#
# Question 4
#
#
q4_file = pd.read_csv('letter-recognition.data', delimiter=',', dtype=None, encoding=None)
q4_file = np.array(q4_file)

# each row represents a letter in the alphabet
q4_targets = np.array(np.identity(26, dtype=float))

# using sigmoid as targets are between 0 -1
mlp_q4 = MLP(16, 16, 26, 0.1, is_sigmoid=True)
print("\n", file=f)
training_data = q4_file[0: 15000, 0:25]  # take the first 4/5 for training
num_of_epochs = 200
for i in range(num_of_epochs):
    error = 0
    error2 = 0
    for j in range(25):
        random_num = random.randint(0, 14999)

        index = ord(training_data[random_num][0]) - 65  # index from ascii
        input_training = training_data[random_num]  # get row
        input_training = np.delete(input_training, [0], 0)  # drop letter
        input_training = np.array(input_training, dtype=float)  # convert to floats
        mlp_q4.train(np.array(input_training, ndmin=2).T, np.array(q4_targets[index], ndmin=2).T)

        error += abs(np.array(q4_targets[index], ndmin=2).T - mlp_q4.feedforward(np.array(input_training, ndmin=2).T))

    if i % 5 == 0 or i == num_of_epochs - 1:
        error = np.sum(error / 26) / 26
        print("Training Error Q4: " + str(error) + "  Epoch:" + str(i), file=f)

# test on remainder question 4
error = 0
testing_data = q4_file[15000:19999, 0:25]
for j in range(15000, 20000):
    random_num = random.randint(0, 4998)

    index = ord(testing_data[random_num][0]) - 65

    input_testing = testing_data[random_num]
    input_testing = np.delete(input_testing, [0], 0)
    input_testing = np.array(input_testing, dtype=float)
    error += abs(np.array(np.array(q4_targets[index], ndmin=2).T - mlp_q4.feedforward(np.array(input_testing, ndmin=2).T)))

error = np.sum(error / 26 ) / 4998  
print("Testing Error Q4: " + str(error), file=f)
