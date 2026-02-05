from Neuron import Neuron, OutputNeuron

class NeuralNetwork:
    def __init__(self, structure):
        #structure is an array containing the number of neurons in each hidden layer

        #construct 2D array, with each array containing the neurons for each layer, the last array containing only the output neuron
        self.layers=[[]]
        for layerSize in structure:
            if self.layers==[[]]:
                for i in range(layerSize):
                    self.layers[0].append(Neuron(2))
                previousLayerSize=layerSize
            else:
                self.layers.append([])
                for i in range(layerSize):
                    self.layers[len(self.layers)-1].append(Neuron(previousLayerSize+1))#extra weight for bias
                previousLayerSize=layerSize

        #add output neuron to the array
        self.layers.append([OutputNeuron(previousLayerSize+1)])
        
        #create 1D array of biases, initialised to zero
        self.biases=[0]*(len(structure)+1)#extra bias for output layer

    def calculate(self, value):
        layerValues=[]
        for layer in self.layers:
            if layerValues==[]:
                layerValues=[[value, self.biases[0]]]#set the first array to the input vector for the first hidden layer
            layerValues.append([])
            for neuron in layer:
                layerValues[len(layerValues)-1].append(neuron.calculate(layerValues[len(layerValues)-2]))
            if len(layerValues)<=len(self.biases):
                layerValues[len(layerValues)-1].append(self.biases[len(layerValues)-1])#add the bias for the next layer

        return layerValues[len(layerValues)-1]