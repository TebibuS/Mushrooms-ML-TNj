import pandas
from pandas import read_csv as read_csv
import os
from numpy import nan
from scipy import stats
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.image as im
# import seaborn
from sklearn import tree


# Get data
DataHeaders = ['classification','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
 'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']


# Adjust file path, can remove getcwd if easier and just put file path in parenthesis. *Must convert to .csv first*
rawData = read_csv(filepath_or_buffer= os.getcwd() + "\\Mushrooms-ML-TNj\\mushroom\\agaricus-lepiota.csv", index_col=False, dtype=str)
# rawData['stalk-root'].replace(to_replace="?", value=nan, inplace=True)
IntMappedData = pandas.DataFrame(columns=rawData.keys(), index=range(len(rawData[rawData.keys()[0]])), dtype=int, data=[[1]*23]*8124)

keyIntMap = {}

BarX = []
BarY = []
for a in rawData.keys():
    tempDict = {}
    keyIntMap[a] = {}
    for b in rawData[a]: 
        if b not in tempDict:
            tempDict[b] = 0
            keyIntMap[a][b] = len(tempDict)
        tempDict[b] = tempDict[b]+1

    BarX.append(tempDict.values())
    BarY.append(tempDict.keys())



# implement int mapping of string values
for i in range(len(rawData.keys())):
    for j in range(len(rawData[rawData.keys()[i]])):
        IntMappedData.loc[j][i] = keyIntMap[rawData.keys()[i]][rawData[rawData.keys()[i]][j]]



# Correlation Matrix
correlationMatrix = IntMappedData.corr()
plt.matshow(correlationMatrix)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=6)
plt.xticks(ticks=range(len(rawData.keys())), labels=rawData.keys(), fontsize=6, rotation=90)
plt.yticks(ticks=range(len(rawData.keys())), labels=rawData.keys(), fontsize=6)
plt.show()

#Box plot graphs of data
be = True
scale = 30
gap = 8
for i in range(len(rawData.keys())):
    plt.subplot2grid((5*scale, 5*scale), ((int(i/5))*scale, int((i%5))*scale), rowspan=int(scale-gap), colspan=int(scale-gap)) 
    plt.bar(x=range(len(BarX[i])), height=BarX[i], align='center', bottom=0)
    plt.xticks(ticks=range(len(BarX[i])), labels=BarY[i])
    plt.title(rawData.keys()[i])
plt.subplots_adjust(left=.025, right=.825, bottom=.2, top=.97, hspace=1, wspace=0)
plt.get_current_fig_manager().full_screen_toggle()
plt.show()