import sys
sys.path.append("../Neural-Network")#import from the parent directory
from NeuralNetwork import *
import pandas as pd
from time import time

nn = fromCSV("MNIST/mnistParameters.csv",leakyReLU,softmax,leakyReLUDerivative,softmaxDerivative)
mnistdf=pd.read_csv("MNIST/mnist_train.csv",header=None)
lastSaved=time()
startTime=lastSaved

while True:
    #save parameters every 100 seconds
    if time()-lastSaved>100:
        nn.saveParameters("MNIST/mnistParameters.csv")
        print("Saved parameters")
        lastSaved=time()
        print(f"Total runtime this session: {round((lastSaved-startTime)/60,2)} minutes")

    #new batch
    batch=mnistdf.sample(n=100)
    #format data into lists usable by the neural network
    inputVectors=[]
    expectedOutputs=[]
    for i in range(len(batch)):
        inputVectors.append(list(batch.iloc[i,1:]))
        expectedOutputs.append([0]*10)
        label=batch.iloc[i,0]
        expectedOutputs[len(expectedOutputs)-1][label]=1

    nn.batchTrain(inputVectors,expectedOutputs,0.1)