from numpy import random, dot
from collections.abc import Callable

class Neuron:
    def __init__(self, numberOfWeights:int, activation:Callable[[float],float], bias:float):
        #numberOfWeights is an integer stating the number of weights necessary
        #activation is a string naming the activation function
        #bias is the initial bias of the neuron
        self.activation=activation
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
        #use activation function to get the final output
        output=self.activation(output)
        return output
    
#exactly the same as any other neuron, but does not use an activation function
class OutputNeuron(Neuron):
    def __init__(self, numberOfWeights:int, bias:float):
        self.bias=bias
        self.weights=[]
        for i in range(numberOfWeights):
            self.weights.append(random.normal(0,1))

    def calculate(self, inputVector:list[float])->float:
        return dot(inputVector,self.weights)+self.bias