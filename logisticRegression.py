"""
file: q2.py
description: CSCI 635 HW1 : Question 2
language: Python
author: Shreya Pramod, sp3045
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    alpha = 0.001

    def __init__(self, iter):
        self.iter = iter
        self.bias = 0

    def dotProduct(self, inputDataPoint, weights):
        eachDataInput = inputDataPoint
        dotProdVal = np.dot(eachDataInput, weights)
        return dotProdVal

    def costFunction(self, X, weights, bias):
        cost = 0
        prod = self.dotProduct(X, weights)
        cost += (prod+bias)
        return cost

    def sigmoidFunction(self, z):
        if (len(z)>=0):
            sigma = 1/(1+np.exp(-z))
        else:
            sigma = np.exp(-z)/(1+np.exp(-z))
        return sigma

    def gradientDescentAlgorithm(self, X, h, Y):
        innerTerm = np.dot(np.transpose(X), (h-Y))
        m = Y.shape[0]
        gradientDescent = (1/m)*innerTerm
        return gradientDescent

    def calculateBias(self, cost, bias):
        b = bias
        for j in range(len(cost)):
            b += self.alpha * (1/len(cost)) * cost[j]
        return b

    def regressionClassifier(self, X, Y):
        self.weight = np.zeros(X.shape[1])
        presentBias = self.bias
        for i in range(self.iter):
            cost = self.costFunction(X, self.weight, self.bias)
            sigmaFn = self.sigmoidFunction(cost)
            gradDesc = self.gradientDescentAlgorithm(X, sigmaFn, Y)
            self.weight -= self.alpha * gradDesc
            self.bias = self.calculateBias(cost, presentBias)
            presentBias = self.bias

    def predict(self, X):
        xDotWeight = np.dot(X, self.weight)
        prediction = self.sigmoidFunction(xDotWeight)
        return prediction

def readData(input1):
    inputData = []
    classData = []

    xValHM, yValHM = [], []
    xValHC, yValHC = [], []

    # HM - 0, HC - 1
    with open(input1, 'r', encoding='UTF-8') as f:
        reader = f.readlines()
        for value in reader:
            value = value.strip("\n")
            content = value.split(",")
            if content[2] == ("HylaMinuta"):
                xValHM.append(content[0])
                yValHM.append(content[1])
                inputData.append([float(content[0]), float(content[1])])
                classData.append(0)
            elif content[2] == "HypsiboasCinerascens":
                xValHC.append(content[0])
                yValHC.append(content[1])
                inputData.append([float(content[0]), float(content[1])])
                classData.append(1)
    return inputData, classData ,xValHM, yValHM, xValHC, yValHC

def arrayConversion(classData, xValHM, yValHM, xValHC, yValHC):

    classArray = np.asarray(classData, dtype=np.float32)
    xValHM1 = np.asarray(xValHM, dtype=np.float32)
    yValHM1 = np.asarray(yValHM, dtype=np.float32)
    xValHC1 = np.asarray(xValHC, dtype=np.float32)
    yValHC1 = np.asarray(yValHC, dtype=np.float32)

    return classArray, xValHM1, yValHM1, xValHC1, yValHC1

def modelAccuracy(modelData, classData, inputFile):
    updatedWeights = regressionModel.sigmoidFunction(regressionModel.dotProduct(modelData, regressionModel.weight)).round()
    if len(updatedWeights) == len(classData):
        print(inputFile+ " has an accuracy of {:.2f}%.".format((updatedWeights == classData).mean() * 100))
    else:
        print("No match")

def modelTrain(input1):
    modelData, classData, xValHM, yValHM, xValHC, yValHC = readData(input1)

    modelData = np.asarray(modelData, dtype=np.float32)
    classArray, xValHM1, yValHM1, xValHC1, yValHC1 = arrayConversion(classData, xValHM, yValHM, xValHC, yValHC)
    regressionModel.regressionClassifier(modelData, classArray)

    x = np.arange(-0.5, 0.5, 0.1)
    w1, w2 = regressionModel.weight

    m = -w1 / w2
    c = regressionModel.bias

    y = m * x + c
    inputData = [1.5, 2.5, -0.5]
    inputDatay, xValHM, yValHM, xValHC, yValHC = arrayConversion(inputData, xValHM, yValHM, xValHC, yValHC)

    modelAccuracy(modelData, classData, input1)

    HM = plt.scatter(xValHM1, yValHM1, marker='o', color='blue', label='HylaMinuta')
    HC = plt.scatter(xValHC1, yValHC1, marker='o', color='red', label='HypsiboasCinerascens')

    plt.title("Scatter Plot for " + input1)
    plt.xlabel("MFCCs_10")
    plt.ylabel("MFCCs_17")
    plt.plot(x, y, color="black", label="Decision Boundary", linewidth=2, linestyle='dashed')

    plt.legend(loc = "upper left")
    plt.show()

if __name__ == '__main__':
    regressionModel = LogisticRegression(1000)
    input1 = "Frogs.csv"
    input2 = "Frogs-subsample.csv"

    modelTrain(input1)
    modelTrain(input2)