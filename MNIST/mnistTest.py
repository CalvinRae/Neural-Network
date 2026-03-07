import sys
sys.path.append("../Neural-Network")#import from the parent directory
from NeuralNetwork import *
import pandas as pd
from time import time

nn = fromCSV("MNIST/mnistParameters.csv",leakyReLU,softmax)
mnistdf=pd.read_csv("MNIST/mnist_test.csv",header=None)

#new batch
batch=mnistdf.sample(n=500)

#format data into lists usable by the neural network
inputVectors=[]
labels=[]
for i in range(len(batch)):
    inputVectors.append(list(batch.iloc[i,1:]/255))#normalise input
    label=batch.iloc[i,0]
    labels.append(label)

successes=0
trials=0
for inputVector, label in zip(inputVectors,labels):
    output=nn.calculate(inputVector)
    if output.index(max(output))==label:
        successes+=1
    trials+=1

print(f"{successes} successes of {trials} trials, % score is {100*successes/trials}")