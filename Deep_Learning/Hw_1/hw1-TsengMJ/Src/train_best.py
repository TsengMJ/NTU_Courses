import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import sys
import os

## 引入讀資料檔
from read import *

### 需要改成讀參數
TRAIN_PATH = sys.argv[1]


# Adagrad
def Adagrad(X, Y, ω, η, Iteration, λ):
    listCost = []
    arrayGradientSum = np.zeros((X.shape[1],1))
    for itera in range(Iteration):
        arrayYHat = np.dot(X, ω)
        arrayLoss = arrayYHat - Y
        arrayCost = np.sqrt(np.mean(arrayLoss**2))

        # save cost function value in process
        listCost.append(arrayCost)

        arrayGradient = (np.dot(np.transpose(X), arrayLoss) / X.shape[0]) + (λ * ω)
        arrayGradientSum += arrayGradient**2
        arraySigma = np.sqrt(arrayGradientSum)
        ω -= η * arrayGradient / arraySigma
               
        if itera % 1000 == 0:
            print("iteration:{}, cost:{} ".format(itera, arrayCost))
    
    return ω, listCost

## 讀入資料
arrayTrainX, arrayTrainY, mean_x, std_x = read(TRAIN_PATH, True)

## Training
# Adagrad
intLearningRate = 1.27
arrayW = np.zeros((arrayTrainX.shape[1],1))  # (163, )
arrayW_ada, listCost_ada = Adagrad(X=arrayTrainX, Y=arrayTrainY, ω=arrayW, η=intLearningRate, Iteration=20000, λ=0)

## Save weight
np.save('../Weight/weight_best.npy', arrayW_ada)
np.save('../Weight/mean_x_best.npy', mean_x)
np.save('../Weight/std_x_best.npy', std_x)
