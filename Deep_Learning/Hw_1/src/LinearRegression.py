import sys
import pandas as pd
import numpy as np

data = pd.read_csv('./train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day*24: (day+1)*24] = raw_data[18*(20*month+day): 18*(20*month+day+1), :]
    month_data[month] = sample

x = np.empty([12*471, 18*9], dtype=float)
y = np.empty([12*471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month*471 + day*24 + hour, :] = month_data[month][:, day*24 + hour : day*24 + hour+9].reshape(1, -1)
            y[month*471 + day*24 + hour, 0] = month_data[month][9, day*24 + hour+9]

mean_x = np.mean(x, axis=0) #18*9
std_x = np.std(x, axis=0)   #18*9
for i in range(len(x)):     #12*471
    for j in range(len(x[0])):  #18*9
        if std_x[j] != 0:
            x[i][j] = (x[i][j]-mean_x[j]) / std_x[j]