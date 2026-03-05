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

def sigmoidDerivative(numbers:list[float])->list[float]:
    output=[]
    for number in numbers:
        output.append((1/(1+exp(-number)))*(1-(1/(1+exp(-number)))))
    return output

def ReLUDerivative(numbers:list[float])->list[float]:
    output=[]
    for number in numbers:
        if number>0:
            output.append(1)
        else:
            output.append(0)
    return output

def leakyReLUDerivative(numbers:list[float])->list[float]:
    output=[]
    for number in numbers:
        if number>0:
            output.append(1)
        else:
            output.append(0.01)
    return output

def linearDerivative(numbers:list[float]|None=None)->1:
    return 1

def softmaxDerivative(numbers:list[float])->list[float]:
    output=softmax(numbers)
    for i in range(len(output)):
        output[i]=output[i]*(1-output[i])
    return output
    
def fromCSV(filePath:str, hiddenActivation:Callable[[float],float], outputActivation:Callable[[float],float], diffHidden:Callable[[list[float]],list[float]]|None=None, diffOutput:Callable[[list[float]],list[float]]|None=None)->NeuralNetwork:
    file=open(filePath,"r")
    storedNN=file.read()
    file.close()
    storedNN=storedNN.split("\n")
    structure=[len(storedNN[0].split(";")[0].split(","))-1]
    for layer in storedNN:
        structure.append(len(layer.split(";")))
    newNN=NeuralNetwork(structure,hiddenActivation,outputActivation,diffHidden=diffHidden,diffOutput=diffOutput)
    newNN.loadParameters(filePath)
    return newNN

class NeuralNetwork:
    def __init__(self, structure:list[int], hiddenActivation:Callable[[list[float]],list[float]], outputActivation:Callable[[list[float]],list[float]], initialBias:float=0, diffHidden:Callable[[list[float]],list[float]]|None=None, diffOutput:Callable[[list[float]],list[float]]|None=None):
        #structure is an array containing the number of neurons in each hidden layer
        #hiddenActivation and outputActivation are callables for the activation functions of the hidden and output layers
        #initialBias is a float which will be used as the initial value for the bias of each layer
        self.hiddenActivation=hiddenActivation
        self.outputActivation=outputActivation
        self.diffHidden=diffHidden
        self.diffOutput=diffOutput

        #construct 2D array, with each array containing the neurons for each layer, the last array containing only the output neurons
        self.layers=[]
        #do not create neurons for the input neurons, the input will be passed as a list to the calculate method
        for i in range(1,len(structure)):
            self.layers.append([])
            for j in range(structure[i]):
                self.layers[len(self.layers)-1].append(Neuron(structure[i-1],initialBias))

    def calculateAll(self, inputVector:list[float])->tuple[list[list[float]],list[list[float]]]:
        #finalValues is a 2d array of floats; each array in finalValues contains the output of each neuron from that layer
        #initialValues is similar, but holds the values of each neuron before applying the activation function
        finalValues=[inputVector]#set the first array to the input vector for the first hidden layer, i.e. the outputs of the input neurons
        initialValues=[inputVector]#there is no activation function on the input layer, so finalValues=initialValues
        for layer in self.layers:
            initialValues.append([])#create new empty array to hold outputs of the current layer
            for neuron in layer:
                initialValues[len(initialValues)-1].append(neuron.calculate(finalValues[len(finalValues)-1]))#add each neuron's output to the array
            if layer is not self.layers[len(self.layers)-1]:#if this is not the output layer
                finalValues.append(self.hiddenActivation(initialValues[len(initialValues)-1]))#use the activation for the hidden layers
            else:
                finalValues.append(self.outputActivation(initialValues[len(initialValues)-1]))#otherwise use the activation for the output layer

        return finalValues, initialValues#returns values of all neurons after and before activation
    
    def calculate(self, inputVector:list[float])->list[float]:
        output = self.calculateAll(inputVector)[0]#take only the final values, after the activation function
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
        for i in range(len(allParameters)):
            allParameters[i]=allParameters[i].split(";")#separate layers into neurons
            for j in range(len(allParameters[i])):
                allParameters[i][j]=allParameters[i][j].split(",")#separate neurons into weights and a bias
                for k in range(len(allParameters[i][j])):
                    allParameters[i][j][k]=float(allParameters[i][j][k])

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

    def train(self,inputVector:list[float],expectedOutput:list[float],learningRate:float)->None:
        activations, initialValues=self.calculateAll(inputVector)
        self.addAdjustments(activations,initialValues,expectedOutput,learningRate)
        self.applyAdjustments(1)

    def batchTrain(self,inputVectors:list[list[float]],expectedOutputs:list[list[float]],learningRate:float)->None:
        for inputVector, expectedOutput in zip(inputVectors,expectedOutputs):
            activations, initialValues=self.calculateAll(inputVector)
            self.addAdjustments(activations,initialValues,expectedOutput,learningRate)
        self.applyAdjustments(len(inputVectors))
        
    def addAdjustments(self,activations:list[list[float]],initialValues:list[list[float]],expectedOutput:list[float],learningRate:float)->None:
        dAdZ=[]#differentiation of each activation function, note that it also includes the input layer
        for layer in initialValues:
            if layer is initialValues[len(initialValues)-1]:
                dAdZ.append(self.diffOutput(layer))
            elif layer is initialValues[0]:
                dAdZ.append([1]*len(layer))#because for the input layer, dAdZ=1
            else:
                dAdZ.append(self.diffHidden(layer))
        dCdA=[]
        for i in range(len(self.layers)):
            dCdA.append([])

        #add adjustments for the output layer
        for i in range(len(expectedOutput)):
            dCdA[len(dCdA)-1].append(2*(activations[len(activations)-1][i]-expectedOutput[i]))#calculate dC/dA for output layer
        for i in range(len(self.layers[len(self.layers)-1])):
            adjustments=[]
            for j in range(len(self.layers[len(self.layers)-1][i].weights)):
                adjustments.append(learningRate*activations[len(activations)-2][j]*dAdZ[len(dAdZ)-1][i]*dCdA[len(dCdA)-1][i])#calculate dC/dW for each weight
            self.layers[len(self.layers)-1][i].addWeightAdjustments(adjustments)#add weight adjustments
            self.layers[len(self.layers)-1][i].addBiasAdjustment(learningRate*dAdZ[len(dAdZ)-1][i]*dCdA[len(dCdA)-1][i])#add bias adjustment

        #iterate through adjustments for hidden layers
        for currentLayer in range(len(self.layers)-2,-1,-1):
            #calculate dC/dA for each neuron
            for currentNeuron in range(len(self.layers[currentLayer])):
                temp=0
                for neuronIndex in range(len(self.layers[currentLayer+1])):
                    temp+=self.layers[currentLayer+1][neuronIndex].weights[currentNeuron]*dAdZ[currentLayer+2][neuronIndex]*dCdA[currentLayer+1][neuronIndex]
                dCdA[currentLayer].append(temp)
            #add adjustments for each neuron
            for currentNeuron in range(len(self.layers[currentLayer])):
                adjustments=[]
                for currentWeight in range(len(self.layers[currentLayer][currentNeuron].weights)):
                    adjustments.append(learningRate*activations[currentLayer][currentWeight]*dAdZ[currentLayer+1][currentNeuron]*dCdA[currentLayer][currentNeuron])
                self.layers[currentLayer][currentNeuron].addWeightAdjustments(adjustments)
                self.layers[currentLayer][currentNeuron].addBiasAdjustment(learningRate*dAdZ[currentLayer+1][currentNeuron]*dCdA[currentLayer][currentNeuron])

    def applyAdjustments(self,count)->None:
        for layer in self.layers:
            for neuron in layer:
                neuron.applyAdjustments(count)