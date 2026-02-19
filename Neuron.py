from numpy import random, dot
from collections.abc import Callable

class Neuron:
    def __init__(self, numberOfWeights:int, bias:float):
        #numberOfWeights is an integer stating the number of weights necessary
        #activation is a string naming the activation function
        #bias is the initial bias of the neuron
        self.bias=bias
        
        #use random weight initialisation, using normal distribution with mean 0, s.d. 1
        self.weights=[]
        for i in range(numberOfWeights):
            self.weights.append(random.normal(0,1))

    def calculate(self,inputVector:list[float])->float:
        #calculate dot product of weights and the values from the previous layer and the bias
        output=dot(inputVector,self.weights)
        #add bias
        output += self.bias
        return output