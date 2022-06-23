import pandas as pd
import matplotlib as plt

dataPath = './data/train.labels'
#dataPath = './data/val.labels'
labelCnt = [0,0,0,0,0,0,0,0,0,0,0,0]
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
f = open(dataPath, 'r')
for line in f.readlines():
    l = int(line)
    labelCnt[l-1] = labelCnt[l-1] + 1

labelArr = list(zip(labels, labelCnt))
df=pd.DataFrame(labelArr,columns=["Labels", "LabelCount"])
#df.plot(kind="bar",figsize=(9,7))
df.plot(x="Labels", y=["LabelCount"], kind="bar",figsize=(9,7))
print('Done')