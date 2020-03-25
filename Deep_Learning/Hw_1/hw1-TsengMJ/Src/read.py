import pandas as pd
import numpy as np

def read(Path, train=True, mean_x=None, std_x=None):
    if (train == True):
        ###---DataProcessing---###
        # 讀進訓練資料
        data = pd.read_csv(Path, encoding='big5')  ## Read data
        data.drop(columns=data.columns[:3], inplace=True) ## Remove first three columns
        data.replace('NR', 0, inplace=True)             ## Replace NR to 0
        data = data.astype(np.float)
        data = data.to_numpy()
        data[data<0] = 0

        # 切分訓練資料
        month_data = {}
        for month in range(12):
            sample = np.empty([18, 480]) 
            # 將資料排程 Row -> 特徵, Col -> 時間
            for day in range(20):
                sample[:, day * 24 : (day + 1) * 24] = data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
            month_data[month] = sample
            
        listTrainX = np.empty([12 * 471, 18 * 9], dtype = float)
        listTrainY = np.empty([12 * 471, 1], dtype = float)
        for month in range(12):
            for day in range(20):
                for hour in range(24):
                    # 每個月只有 20 天的訓練資料，後面幾個小時無法使用
                    if day == 19 and hour > 14:
                        continue
                    listTrainX[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
                    listTrainY[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value

        # 正規化訓練資料
        mean_x = np.mean(listTrainX, axis=0) #18 * 9 
        std_x = np.std(listTrainX, axis=0) #18 * 9 

        for i in range(len(listTrainX)): #12 * 471
            for j in range(len(listTrainX[0])): #18 * 9 
                if std_x[j] != 0:
                    listTrainX[i][j] = (listTrainX[i][j] - mean_x[j]) / std_x[j]

        arrayTrainX = np.array(listTrainX)
        arrayTrainY = np.array(listTrainY)
        arrayTrainX = np.insert(arrayTrainX, arrayTrainX.shape[1], values=1, axis=1) # (5652, 163)

        return arrayTrainX, arrayTrainY, mean_x, std_x
    

    else: 
    # 讀進測試資料
        data = pd.read_csv(Path, header = None, encoding = 'big5')
        data = data.iloc[:, 2:]
        data[data == 'NR'] = 0
        data = data.astype(np.float)
        data = data.to_numpy()

        listTestX = np.empty([240, 18*9], dtype = float)
        for i in range(240):
            listTestX[i, :] = data[18 * i: 18* (i + 1), :].reshape(1, -1)
            
        listTestX[listTestX < 0] = 0

            
        for i in range(len(listTestX)):
            for j in range(len(listTestX[0])):
                if std_x[j] != 0:
                    listTestX[i][j] = (listTestX[i][j] - mean_x[j]) / std_x[j]
                    
        arrayTestX = np.array(listTestX)

        # 增加bias項
        arrayTestX = np.insert(arrayTestX, arrayTestX.shape[1], values=1, axis=1) # (5652, 163)

        return arrayTestX