import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import sys
import os

## 引入讀資料檔
from read import *

### 需要改成讀參數
TEST_PATH   = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
WEIGHT_PATH = (sys.argv[3]==0) and './Weight/weight.npy' or './Weight/weight_best.npy'
MEAN_PATH   = (sys.argv[3]==0) and './Weight/mean_x.npy' or './Weight/mean_x_best.npy'
STD_PATH    = (sys.argv[3]==0) and './Weight/std_x.npy'  or './Weight/std_x_best.npy'

mean_x = np.load(MEAN_PATH)
std_x  = np.load(STD_PATH)


## Load data
arrayTestX = read(TEST_PATH, False, mean_x, std_x)

## Load Weight
w = np.load(WEIGHT_PATH)
ans_y = np.dot(arrayTestX, w)
ans_y = np.round(ans_y)

## Store result
with open(OUTPUT_PATH, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)