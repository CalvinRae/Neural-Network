from Neuron import *
from numpy import exp
from collections.abc import Callable

def sigmoid(number:float)->float:
    return 1/(1+exp(-number))

def ReLU(number:float)->float:
    if number<0:
        return 0
    else:
        return number
    
def Linear(number:float)->float:
    return number

def leakyReLU(number:float)->float:
    if number<0:
        return 0.01*number
    else:
        return number

def sigmoidDerivative(number:float)->float:
    return sigmoid(number)*(1-sigmoid(number))

def ReLUDerivative(number:float)->float:
    if number>0:
        return number
    else:
        return 0
    
def linearDerivative(number:float | None = None)->float:
    return 1

def leakyReLUDerivative(number:float)->float:
    if number<0:
        return 0.01
    else:
        return 1

class NeuralNetwork:
    def __init__(self, structure:list[int], activation:Callable[[float],float], initialBias:float=0):
        #structure is an array containing the number of neurons in each hidden layer
        #activation is a string naming the activation function
        #initialBias is a float which will be used as the initial value for the bias of each layer

        #construct 2D array, with each array containing the neurons for each layer, the last array containing only the output neuron
        self.layers=[]
        index=1#do not create neurons for the input neurons, the input will be passed as a list to the calculate method
        while index<=len(structure)-1:
            self.layers.append([])
            if index<len(structure)-1:
                for i in range(structure[index]):
                    self.layers[len(self.layers)-1].append(Neuron(structure[index-1],activation,initialBias))
            else:
                for i in range(structure[index]):
                    self.layers[len(self.layers)-1].append(OutputNeuron(structure[index-1],initialBias))
            index+=1

    def calculate(self, inputVector:list[float])->float:
        #layerValues is a 2d array of floats; each array in layerValues contains the output of each neuron from that layer
        layerValues=[inputVector]#set the first array to the input vector for the first hidden layer, i.e. the outputs of the input neurons
        for layer in self.layers:
            layerValues.append([])#create new empty array to hold outputs of the current layer
            for neuron in layer:
                layerValues[len(layerValues)-1].append(neuron.calculate(layerValues[len(layerValues)-2]))#add each neuron's output to the array

        return layerValues[len(layerValues)-1]#returns values of the output neurons
    
    def saveWeights(self, filePath:str)->None:
        file=open(filePath,"w")
        for layer in self.layers:
            for neuron in layer:
                for weight in neuron.weights:
                    file.write(str(weight))
                    if weight is not neuron.weights[len(neuron.weights)-1]:#place a comma after every weight but the last one (in between all weights)
                        file.write(",")
                if neuron is not layer[len(layer)-1]:
                    file.write(";")#place a semicolon to separate weights for different neurons
            if layer is not self.layers[len(self.layers)-1]:
                file.write("\n")#place a linebreak to separate neurons for different layers
        file.close()

    def loadWeights(self, filePath:str)->None:
        #read csv
        file=open(filePath,"r")
        allWeights=file.read()
        file.close()

        #take the string from the file and format it into a useful data structure
        allWeights=allWeights.split("\n")#separate all weights into layers
        i=0
        while i<len(allWeights):
            allWeights[i]=allWeights[i].split(";")#separate layers into neurons
            j=0
            while j<len(allWeights[i]):
                allWeights[i][j]=allWeights[i][j].split(",")#separate neurons into weights
                k=0
                while k<len(allWeights[i][j]):
                    allWeights[i][j][k]=float(allWeights[i][j][k])
                    k+=1
                j+=1
            i+=1

        #check that the stored weights and this neural network have the same numbers of layers and neurons in each layer
        sameStructure=True
        if len(self.layers)!=len(allWeights):#check that number of layers are equal
            sameStructure=False
        else:
            for layer, layerWeights in zip(self.layers,allWeights):
                if len(layer)!=len(layerWeights):#check that number of neurons in each layer are equal
                    sameStructure=False
                    break
        
        #set each neuron's weights to the stored weights
        if sameStructure:
            for layer, layerWeights in zip(self.layers,allWeights):
                for neuron, neuronWeights in zip(layer, layerWeights):
                    neuron.weights=neuronWeights