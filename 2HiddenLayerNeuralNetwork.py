import numpy
import scipy.special

class neuralNetwork:
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes
        self.weights_input_hidden1 = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden1_nodes, self.input_nodes))
        self.weights_hidden1_hidden2 = numpy.random.normal(0.0, pow(self.hidden1_nodes, -0.5), (self.hidden2_nodes, self.hidden1_nodes))
        self.weights_hidden2_output = numpy.random.normal(0.0, pow(self.hidden2_nodes, -0.5), (self.output_nodes, self.hidden2_nodes))
        self.learning_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden1_inputs = numpy.dot(self.weights_input_hidden1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.weights_hidden1_hidden2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = numpy.dot(self.weights_hidden2_output, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        # entiendo hasta aca
        output_errors = targets - final_outputs
        hidden2_errors = numpy.dot(self.weights_hidden2_output.T, output_errors)
        hidden1_errors = numpy.dot(self.weights_hidden2_output.T, output_errors)
        self.weights_hidden2_output += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))
        self.weights_hidden1_hidden2 += self.learning_rate * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)), numpy.transpose(hidden1_outputs))
        self.weights_input_hidden1 += self.learning_rate * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs))
        pass
    
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden1_inputs = numpy.dot(self.weights_input_hidden1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.weights_hidden1_hidden2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = numpy.dot(self.weights_hidden2_output, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)
        # print(self.weights_hidden2_output)
        return final_outputs

input_nodes = 784
hidden1_nodes = 100
hidden2_nodes = 100
output_nodes = 10

learning_rate = 0.01

n = neuralNetwork( input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10

print("Training...")
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []
print("Testing...")

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size * 100,'%')