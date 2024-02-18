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

def FeatureSeparation(rawDataFilePath, printToCSV=False):
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

    Target = rawData[rawData.keys()[0]].replace(to_replace={'p': 1,
                                                            'e': 0})
    rawData.drop('classification', axis='columns')
    Data = pandas.DataFrame(index=range(len(rawData[rawData.keys()[0]])), columns=FeatureMap, dtype=bool).replace(to_replace=True, value=False)
    Data = Data.astype(dtype=int, copy=True)

    for col in range(len(FeatureListToTokenMap)):
        for ind in range(len(rawData[rawData.columns[0]])):
            if(rawData.at[ind,FeatureListToKeyMap[col]] == FeatureListToTokenMap[col]):
                Data.at[ind,FeatureMap[col]] = 1
    if printToCSV:
        df = pandas.DataFrame(Data)
        df.insert(loc=len(Data.columns), column='Target', value=Target)
        df.to_csv(path_or_buf=os.getcwd()+"\\mushroom\\FeatureSelectedData.csv")

    return Data, Target, FeatureMap, FeatureCounts, FeatureRawValues



def ShowGraphs():
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




from sklearn import metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics


def RFRunner(rawData, filePath=None):
    if not (filePath == None):
        rawData = read_csv(filepath_or_buffer=filePath, dtype=int, na_filter=False, index_col=0)
    X = rawData[rawData.columns[:len(rawData.columns)-1]]
    y = rawData.get(rawData.columns[len(rawData.columns)-1])
    print(y.shape)
    RandomForestClassifier().get_params()

    n_estimators_list = list(range(10,len(X.columns),20))
    criterion_list = ['gini', 'entropy']
    max_depth_list = [2, 3, 4, 5, 7, 9]
    temp = list(range(10,int(len(X.columns)),5))
    max_depth_list.append(temp)
    min_samples_split_list = [x/1000 for x in list(range(5, len(X.columns)%10+1, 10))]
    min_samples_leaf_list = [x/1000 for x in list(range(5, len(X.columns)%10+1, 10))]
    max_features_list = ['sqrt', 'log2']

    params_grid = {
        'n_estimators': n_estimators_list,
        'criterion': criterion_list,
        'max_depth': max_depth_list,
        'min_samples_split': min_samples_split_list,
        'min_samples_leaf': min_samples_leaf_list,
        'max_features': max_features_list
    }

    num_combinations = 1
    for k in params_grid.keys(): num_combinations *= len(params_grid[k])

    print('Number of combinations = ', num_combinations)
    params_grid
    print("IGNORE FOLLOWING ERRORS (NO IMPACT ON PROGRAM), OR FIND CAUSE (RELATED TO ShowGraphs())")

    def my_roc_auc_score(model, X, y): return metrics.roc_auc_score(y, model.predict(X))
    model_rf = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced'),
                            param_grid=params_grid,
                            n_jobs=3,
                            cv=3,
                            scoring=my_roc_auc_score,
                            return_train_score=True,)
    # model_rf = RandomizedSearchCV(estimator=RandomForestClassifier(class_weight='balanced'),
    #                               param_distributions=params_grid,
    #                               n_iter=50,
    #                               cv=3,
    #                               scoring=my_roc_auc_score,
    #                               return_train_score=True,
    #                               verbose=2)

    model_rf.fit(X,y)

    print(model_rf.best_params_)

    df_cv_results = pandas.DataFrame(model_rf.cv_results_)
    df_cv_results = df_cv_results[['rank_test_score','mean_test_score','mean_train_score',
                            'param_n_estimators', 'param_min_samples_split','param_min_samples_leaf',
                            'param_max_features', 'param_max_depth','param_criterion']]
    df_cv_results.sort_values('rank_test_score', inplace=True)
    print(df_cv_results['mean_test_score'])

    print(df_cv_results[:20])
   








# Data, Target, FeatureMap, FeatureCounts, FeatureRawValues = FeatureSeparation(rawDataFilePath, printToCSV=True)
# ShowGraphs()


featureSelectedDataFilePath = os.getcwd() + "\\mushroom\\FeatureSelectedData.csv"
rawData = read_csv(filepath_or_buffer=featureSelectedDataFilePath, dtype=int, na_filter=False, index_col=0)

RFRunner(rawData=rawData)



