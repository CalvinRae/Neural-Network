from __future__ import annotations
from Neuron import *
from numpy import exp
from collections.abc import Callable

def sigmoid(numbers:list[float])->list[float]:
    output=[]
    for number in numbers:
        output.append(1/(1+exp(-number)))
    return output

def ReLU(numbers:list[float])->list[float]:
    output=[]
    for number in numbers:
        if number<0:
            output.append(0)
        else:
            output.append(number)
    return output

def leakyReLU(numbers:list[float])->list[float]:
    output=[]
    for number in numbers:
        if number<0:
            output.append(0.01*number)
        else:
            output.append(number)
    return output
    
def linear(numbers:list[float])->list[float]:
    return numbers

def softmax(numbers:list[float])->list[float]:
    sumExp=0
    for number in numbers:
        sumExp+=exp(number)
    output=[]
    for number in numbers:
        output.append(exp(number)/sumExp)
    return output

def sigmoidDerivative(number:float)->float:
    return sigmoid(number)*(1-sigmoid(number))

def ReLUDerivative(number:float)->float:
    if number>0:
        return 1
    else:
        return 0

def leakyReLUDerivative(number:float)->float:
    if number<0:
        return 0.01
    else:
        return 1
    
def fromCSV(filePath:str, hiddenActivation:Callable[[float],float], outputActivation:Callable[[float],float])->NeuralNetwork:
    file=open(filePath,"r")
    storedNN=file.read()
    file.close()
    storedNN=storedNN.split("\n")
    structure=[len(storedNN[0].split(";")[0].split(","))-1]
    for layer in storedNN:
        structure.append(len(layer.split(";")))
    newNN=NeuralNetwork(structure,hiddenActivation,outputActivation)
    newNN.loadParameters(filePath)
    return newNN

class NeuralNetwork:
    def __init__(self, structure:list[int], hiddenActivation:Callable[[list[float]],list[float]], outputActivation:Callable[[list[float]],list[float]], initialBias:float=0):
        #structure is an array containing the number of neurons in each hidden layer
        #hiddenActivation and outputActivation are callables for the activation functions of the hidden and output layers
        #initialBias is a float which will be used as the initial value for the bias of each layer
        self.hiddenActivation=hiddenActivation
        self.outputActivation=outputActivation

        #construct 2D array, with each array containing the neurons for each layer, the last array containing only the output neurons
        self.layers=[]
        index=1#do not create neurons for the input neurons, the input will be passed as a list to the calculate method
        while index<=len(structure)-1:
            self.layers.append([])
            for i in range(structure[index]):
                self.layers[len(self.layers)-1].append(Neuron(structure[index-1],initialBias))
            index+=1

    def calculateAll(self, inputVector:list[float])->list[list[float]]:
        #layerValues is a 2d array of floats; each array in layerValues contains the output of each neuron from that layer
        layerValues=[inputVector]#set the first array to the input vector for the first hidden layer, i.e. the outputs of the input neurons
        for layer in self.layers:
            layerValues.append([])#create new empty array to hold outputs of the current layer
            for neuron in layer:
                layerValues[len(layerValues)-1].append(neuron.calculate(layerValues[len(layerValues)-2]))#add each neuron's output to the array
            if layer is not self.layers[len(self.layers)-1]:#if this is not the output layer
                layerValues[len(layerValues)-1]=self.hiddenActivation(layerValues[len(layerValues)-1])#use the activation for the hidden layers
            else:
                layerValues[len(layerValues)-1]=self.outputActivation(layerValues[len(layerValues)-1])#otherwise use the activation for the output layer

        return layerValues#returns values of all neurons
    
    def calculateOutput(self, inputVector:list[float])->list[float]:
        output = self.calculateAll(inputVector)
        return output[len(output)-1]#get only the values for the output neurons
    
    def saveParameters(self, filePath:str)->None:
        file=open(filePath,"w")
        for layer in self.layers:
            for neuron in layer:
                for weight in neuron.weights:
                    file.write(str(weight))
                    file.write(",")
                file.write(str(neuron.bias))#store the bias at the end of the weights
                if neuron is not layer[len(layer)-1]:#don't place a semicolon after the last neuron
                    file.write(";")#place a semicolon to separate weights for different neurons
            if layer is not self.layers[len(self.layers)-1]:
                file.write("\n")#place a linebreak to separate neurons for different layers
        file.close()

    def loadParameters(self, filePath:str)->None:
        #read csv
        file=open(filePath,"r")
        allParameters=file.read()
        file.close()

        #take the string from the file and format it into a useful data structure
        allParameters=allParameters.split("\n")#separate all weights into layers
        i=0
        while i<len(allParameters):
            allParameters[i]=allParameters[i].split(";")#separate layers into neurons
            j=0
            while j<len(allParameters[i]):
                allParameters[i][j]=allParameters[i][j].split(",")#separate neurons into weights and a bias
                k=0
                while k<len(allParameters[i][j]):
                    allParameters[i][j][k]=float(allParameters[i][j][k])
                    k+=1
                j+=1
            i+=1

        #check that the stored parameters and this neural network have the same numbers of layers and neurons in each layer
        sameStructure=True
        if len(self.layers)!=len(allParameters):#check that number of layers are equal
            sameStructure=False
        else:
            for layer, layerParameters in zip(self.layers,allParameters):
                if len(layer)!=len(layerParameters):#check that number of neurons in each layer are equal
                    sameStructure=False
                    break
        
        #set each neuron's parameters to the stored parameters
        if sameStructure:
            for layer, layerParameters in zip(self.layers,allParameters):
                for neuron, neuronParameters in zip(layer, layerParameters):
                    neuron.bias=neuronParameters.pop()
                    neuron.weights=neuronParameters

#TODO:
#-implement softmax function's derivative
#-NOTE: we will only differentiate the softmax function with respect to the same value that it was originally applied to, so its derivative will always be S(1-S),
#where S is the output of the softmax function
#-change derivative functions to act on layer outputs rather than single neuron outputs
#-implement backpropagation, layer by layer, from the output layer backwards
#-iterate over each hidden layer once for every neuron in the next layer (calculate effect on cost via each neuron)