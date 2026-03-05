from numpy import random, dot
from collections.abc import Callable

class Neuron:
    def __init__(self, numberOfWeights:int, bias:float):
        #numberOfWeights is an integer stating the number of weights necessary
        #activation is a string naming the activation function
        #bias is the initial bias of the neuron
        self.bias=bias
        self.weightAdjustments=None
        self.biasAdjustment=None
        
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
    
    def addWeightAdjustments(self,newWeightAdjustments:list[float])->None:
        if self.weightAdjustments==None:
            self.weightAdjustments=newWeightAdjustments
        else:
            for i in range(len(self.weightAdjustments)):
                self.weightAdjustments[i]+=newWeightAdjustments[i]
    
    def addBiasAdjustment(self,newBiasAdjustment:float)->None:
        if self.biasAdjustment==None:
            self.biasAdjustment=newBiasAdjustment
        else:
            self.biasAdjustment+=newBiasAdjustment

    def applyAdjustments(self,count):
        for i in range(len(self.weights)):
            self.weights[i]-=self.weightAdjustments[i]/count
        self.bias-=self.biasAdjustment/count
        self.weightAdjustments=None
        self.biasAdjustment=None