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

# gradient decent
def GD(X, Y, ω, η, Iteration, λ): 
    listCost = []
    for itera in range(Iteration):
        arrayYHat = X.dot(ω)
        arrayLoss = arrayYHat - Y
        arrayCost = np.sqrt(np.mean(arrayLoss**2))
        listCost.append(arrayCost)

        arrayGradient = (X.T.dot(arrayLoss) / X.shape[0]) + (λ * ω)
        ω -= η * arrayGradient
        if itera % 1000 == 0:
            print("iteration:{}, cost:{} ".format(itera, arrayCost))
    return ω, listCost

## 讀入資料
arrayTrainX, arrayTrainY, mean_x, std_x = read(TRAIN_PATH, True)

## Training
intLearningRate = 0.05
arrayW = np.zeros((arrayTrainX.shape[1],1))  # (163, )
arrayW_gd, listCost_gd = GD(X=arrayTrainX, Y=arrayTrainY, ω=arrayW, η=intLearningRate, Iteration=20000, λ=0)

## Save weight
np.save('../Weight/weight.npy', arrayW_gd)
np.save('../Weight/mean_x.npy', mean_x)
np.save('../Weight/std_x.npy', std_x)