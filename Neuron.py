from numpy import random, dot, exp

class Neuron:
    def __init__(self, numberOfWeights):
        #use random weight initialisation, using normal distribution with mean 0, s.d. 1
        self.weights=[]
        for i in range(numberOfWeights):
            self.weights.append(random.normal(0,1))

    def calculate(self,inputVector):
        #calculate dot product of weights and the values from the previous layer and the bias
        output=dot(inputVector,self.weights)

        #use sigmoid function as activation function
        output=1/(1+exp(-output))
