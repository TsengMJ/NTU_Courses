# Predict PM2.5 

## Data 

* 本次作業使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。

* train.csv：每個月前20天的完整資料，總共有5652筆train data，每筆資料維度為162。

* test.csv：從剩下的資料當中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。

![](./Output/TrainProcess.png)

![](./Output/Compare.png)
