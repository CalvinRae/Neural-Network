from numpy import random, dot, exp
from functools import partial

class Neuron:
    def __init__(self, numberOfWeights, activation):
        #use random weight initialisation, using normal distribution with mean 0, s.d. 1
        #activation is a string naming the activation function
        self.activation=activation

        self.weights=[]
        for i in range(numberOfWeights):
            self.weights.append(random.normal(0,1))

    def calculate(self,inputVector):
        #calculate dot product of weights and the values from the previous layer and the bias
        output=dot(inputVector,self.weights)

        #use partial function to call the activation function
        output=self.activation(output)

        return output
    
#exactly the same as any other neuron, but does not use an activation function
class OutputNeuron(Neuron):
    def __init__(self, numberOfWeights):
        #random weight initialisation, no activation function
        self.weights=[]
        for i in range(numberOfWeights):
            self.weights.append(random.normal(0,1))

    def calculate(self, inputVector):
        return dot(inputVector,self.weights)