import numpy
import scipy.special
# numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
# numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

data = numpy.random.normal(0.0, pow(784, -0.5), (2, 3))
data2 = numpy.random.normal(0.0, pow(784, -0.5), (1, 2))
print(data)
print(data2)