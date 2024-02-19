# Block 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


#Block 4

import os
df_train = pd.read_csv(filepath_or_buffer=os.getcwd() + "\\mushroom\\FeatureSelectedData.csv", dtype=int, na_filter=False, index_col=0)

var_columns = [c for c in df_train.columns if c not in ['Target']]
X = df_train.loc[:,var_columns]
y = df_train.loc[:,'Target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

#Block 5

def tree_training(max_leaf_nodes, X_train, y_train, X_valid, y_valid):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, class_weight='balanced')
    model_tree.fit(X_train, y_train)
    
    y_train_pred = model_tree.predict(X_train)
    y_valid_pred = model_tree.predict(X_valid)
    
    auc_train = metrics.roc_auc_score(y_train, y_train_pred)
    auc_valid = metrics.roc_auc_score(y_valid, y_valid_pred)
    
    print("Nodes:{}, Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(max_leaf_nodes,
                                                                     auc_train,
                                                                     auc_valid,
                                                                     auc_train-auc_valid))
          

# Run few iterations to find which max_tree_nodes works best
for i in range(2, 20):
    tree_training(i, X_train, y_train, X_valid, y_valid)


#Block 6
    
kfold = KFold(5, shuffle=True, random_state=1)

for idx_train, idx_valid in kfold.split(df_train):
    X_train = df_train.loc[idx_train, var_columns]
    y_train = df_train.loc[idx_train, 'Target']
    
    X_valid = df_train.loc[idx_valid, var_columns]
    y_valid = df_train.loc[idx_valid, 'Target']
    
    # Try 10 leaf nodes, we saw lot of leaf nodes don't increase performance
    print("Iteration Starts")
    for i in range(2, 16):
        tree_training(i, X_train, y_train, X_valid, y_valid)
    
    print("Iteration Ends\n-----------------------")


#Block 7
    
# CV function requires a scorer of this form
def cv_roc_auc_scorer(model, X, y): return metrics.roc_auc_score(y, model.predict(X))

# Loop through multiple values of max_leaf_nodes to find best parameter
for num_leaf_node in range(2,16):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=num_leaf_node, class_weight='balanced')
    kfold_scores = cross_validate(model_tree,
                                  X,
                                  y,
                                  cv=5,
                                  scoring=cv_roc_auc_scorer,
                                  return_train_score=True)

    # Find average train and test score
    train_auc_avg = np.mean(kfold_scores['train_score'])
    test_auc_avg = np.mean(kfold_scores['test_score'])

    print("Nodes:{}, Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(num_leaf_node,
                                                                     train_auc_avg,
                                                                     test_auc_avg,
                                                                     train_auc_avg-test_auc_avg))

#Block 8

model_tree = DecisionTreeClassifier(max_leaf_nodes=8, class_weight='balanced')
model_tree.fit(X, y)


#Block 9

plt.figure(figsize=(20,10))

plot_tree(model_tree,
          feature_names=var_columns,
          class_names = ['0','1'],
          rounded=True,
          filled=True)

plt.show()


#Block 10

y_pred = model_tree.predict(X)

fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
metrics.auc(fpr, tpr)


#Block 11

zeros_probs = [0 for _ in range(len(y))]
fpr_zeros, tpr_zeros, _ = metrics.roc_curve(y, zeros_probs)

# Plot the roc curve for the model
plt.plot(fpr_zeros, tpr_zeros, linestyle='--', label='No Model')
plt.plot(fpr, tpr, marker='.', label='Model')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add legend
plt.legend()

plt.show()

#Block 12
df_test = pd.read_csv(os.getcwd() + "\\mushroom\\FeatureSelectedData.csv")
TRG_ = df_test['Target']

X_test = df_test.loc[:, var_columns]
y_test_pred  = model_tree.predict(X_test)
df_sample_subm = pd.read_csv(os.getcwd() + "\\mushroom\\FeatureSelectedData.csv")


df_sample_subm['Target'] = y_test_pred


output_dir = os.getcwd() + "\\mushroom"
df_sample_subm.to_csv(output_dir + '/01_tree_scores.csv', index=False)
#Block 13


