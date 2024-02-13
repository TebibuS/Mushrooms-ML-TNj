import pandas
from pandas import read_csv as read_csv
import os
from numpy import nan
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn import tree


# Get data
DataHeaders = ['classification','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
 'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']


# Adjust file path, can remove getcwd if easier and just put file path in parenthesis. *Must convert to .csv first*
rawData = read_csv(filepath_or_buffer= os.getcwd() + "\\mushroom\\agaricus-lepiota.csv", index_col=False, dtype=str)
FormattedData = pandas.DataFrame()


for a in rawData.keys():
    for b in range(len(rawData.get(a))):
        FormattedData.at[b, a] = ord(bytearray(rawData.at[b, a], encoding='UTF-8'))
FormattedData = FormattedData.astype(int)

Features = FormattedData.get(rawData.keys()[1:])
Classification = FormattedData.get(rawData.keys()[0])

BarX = []
BarY = []
for a in rawData.keys():
    tempDict = {}
    for b in rawData[a]:
        tempDict[b] = 0
    for b in rawData[a]:
        tempDict[b] = tempDict[b]+1

    BarX.append(tempDict.values())
    BarY.append(tempDict.keys())
    
print(BarX[0])
print(BarY[0])

#Plot Graphs of data
for i in range(len(rawData.keys())):
    plt.subplot2grid((5, 5), (int(i/5), int(i%5))) 
    plt.bar(x=range(len(BarX[i])), height=BarX[i], align='center', bottom=range(len(BarY[i])))
    plt.xticks(ticks=range(len(BarX[i])), labels=BarY[i])
    plt.title(rawData.keys()[i])
plt.show()
