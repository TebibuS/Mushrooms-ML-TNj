import pandas
from pandas import read_csv as read_csv
import os
from numpy import nan
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.image as im
# import seaborn
from sklearn import tree


# Get data
# DataHeaders = ['classification','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
#  'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
rawDataFilePath= os.getcwd() + "\\mushroom\\agaricus-lepiota.csv"

def FeatureSeparation(rawDataFilePath):
# Adjust file path, can remove getcwd if easier and just put file path in parenthesis. *Must convert to .csv first*
    rawData = read_csv(filepath_or_buffer=rawDataFilePath, index_col=False, dtype=str)
    # rawData['stalk-root'].replace(to_replace="?", value=nan, inplace=True)

    # Map out possible feature choices
    FeatureCounts = []
    FeatureRawValues = []
    for a in rawData.keys():
        tempDict = {}
        for b in rawData[a]: 
            if b not in tempDict:
                tempDict[b] = 0
            tempDict[b] = tempDict[b]+1
        FeatureCounts.append(tempDict.values())
        FeatureRawValues.append(list(tempDict.keys()))

    # Create the Feature Indexes
    FeatureMap = [] #Column Titles
    FeatureListToKeyMap = []
    FeatureListToTokenMap = []
    for i in range(len(FeatureRawValues))[1:]:
        for j in FeatureRawValues[i]:
            FeatureMap.append(f"{rawData.keys()[i]}: {j}")
            FeatureListToKeyMap.append(rawData.keys()[i])
    for i in range(len(FeatureRawValues))[1:]:
        for j in range(len(FeatureRawValues[i])):
            FeatureListToTokenMap.append(FeatureRawValues[i][j])

    Target = rawData[rawData.keys()[0]]
    rawData.drop('classification', axis='columns')
    Data = pandas.DataFrame(index=range(len(rawData[rawData.keys()[0]])), columns=FeatureMap, dtype=bool).replace(to_replace=True, value=False)

    for col in range(len(FeatureListToTokenMap)):
        for ind in range(len(rawData[rawData.columns[0]])):
            if(rawData.at[ind,FeatureListToKeyMap[col]] == FeatureListToTokenMap[col]):
                Data.at[ind,FeatureMap[col]] = True

    return Data, Target, FeatureMap, FeatureCounts, FeatureRawValues

# Data.to_csv(path_or_buf=os.getcwd() + "\\mushroom\\FeatureSelected.csv")

Data, Target, FeatureMap, FeatureCounts, FeatureRawValues = FeatureSeparation(rawDataFilePath)




# Correlation Matrix
correlationMatrix = Data.corr()
plt.matshow(correlationMatrix)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=6)
plt.xticks(ticks=range(len(FeatureMap)), labels=FeatureMap, fontsize=6, rotation=90)
plt.yticks(ticks=range(len(FeatureMap)), labels=FeatureMap, fontsize=6)
plt.show()

#Box plot graphs of data
gridSize = int(int(len(FeatureCounts)**.5)+1)
for j in range(gridSize):
    for l in range(gridSize):
        i = gridSize*j +l
        try:
            FeatureCounts[i]
            plt.subplot2grid((gridSize, gridSize), (j, l), rowspan=1, colspan=1) 
            plt.bar(x=range(len(FeatureCounts[i])), height=FeatureCounts[i], align='center', bottom=0)
            plt.xticks(ticks=range(len(FeatureCounts[i])), labels=FeatureRawValues[i])
            plt.title(Data.keys()[i])
        except IndexError:
            plt.cla()
plt.subplots_adjust(left=.025, right=.79, bottom=.25, top=.97, hspace=.9, wspace=.6)
plt.get_current_fig_manager().full_screen_toggle()
plt.show()




# from sklearn.feature_selection import RFECV 
# # import !gridsearch! 
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
# import multiprocessing as mp
# import concurrent.futures as conc


# # GridSearch: Look up


# outputs = {}

# X = IntMappedData.drop(["classification"], axis = 'columns')
# y = IntMappedData["classification"]

# def func(X, y, pid):
#     print(pid)
#     regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=2)
#     feature_selector = RFE(regressor, step=2)

#     fit = feature_selector.fit(X,y)

#     ranking = feature_selector.ranking_
#     optimal_feature_count = feature_selector.n_features_
#     outputs[pid] = (optimal_feature_count, ranking)

# # with conc.ThreadPoolExecutor(max_workers=1) as executor:
# #     future = executor.submit(processFunction, X, y)
# #     print(future.result())

# proc = []
# for ii in range(5)[1:]:
#     for jj in range(5)[2:]:
#         for kk in range(5)[1:]:
#             for ll in range(22)[1:]:
#                 p = mp.Process(target=func, args=(X, y, (ii, jj, kk, ll), ii, jj, kk, ll))
#                 p.run()
#                 proc.append(p)
# print("Here!")
# for i in proc:
#     try:
#         if(not i.is_alive()):
#             i.join()
#     except AssertionError:
#         pass
# print("Finished!")

# for i in outputs.keys():
#     print(outputs[i])

# # # X_new = X.loc[:, feature_selector.get_support()]

# # # plt.plot(range(1, len(fit.ranking_) + 1), fit., marker = "o")
# # # plt.ylabel("Model Score")
# # # plt.xlabel("Number of Features")
# # # plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.ranking_), 4)})")
# # # plt.tight_layout() 
# # # plt.show()