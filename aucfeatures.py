import pandas as pd
# FIXME: Pandas module är version 1.05, du har ver 1.04
from sklearn.metrics import roc_curve, auc
# FIXME: Matplotlib är version 3.2.2, du har version 3.2.1
import matplotlib.pyplot as plt
# FIXME: IPython module ör version 7.16.1, du har version 7.15.0
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.model_selection import StratifiedShuffleSplit

# ----- iteration of models developed using 91 features - 1 in each loop. Plot curve: AUC vs numbers of features.

path = os.getcwd()
item = "/Population_full.csv"

df = pd.read_csv(path+item,';')

item_features = "/meanValueFolderData.xlsx"

meanValueDf = pd.read_excel(path+item_features)

list_of_features = meanValueDf['variables'].to_list()

features = df.drop(["ID", "Ålder", "Daysinadmission", "Död", "Daystodeath", "MortalityInhospital", "Mortality1day", "Mortality7days", "Mortality30days", "Mortality1Year", "Survival7days", "Prio", "Kön", "severe_sepsis"], axis=1).columns
predict = 'Mortality30days'

clf = BalancedRandomForestClassifier(n_estimators=100)


def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""

    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # f, ax = plt.subplots(figsize=(14, 10))

    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # Plot the luck line.
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    print("mean_auc",mean_auc)
    return mean_auc

def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

print("++++++++++++++++++++++++++++++++++++++++++++++++++")


cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8, random_state=0)

# cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []



# Skapa dataframe för mean auc vs number of features
meanAucDf = pd.DataFrame(columns=['numbers of features', 'mean auc'])




max_x_scale = len(features)
count = 0
for index in list_of_features:
    # if index == 'Audiosensitivity':
    #    break

    X = df.loc[:, features]
    y = df[predict]
    print("number of features", len(features))
    for (train, test), i in zip(cv.split(X, y), range(10)):
        clf.fit(X.iloc[train], y.iloc[train])
        _, _, auc_score_train = compute_roc_auc(train)
        fpr, tpr, auc_score = compute_roc_auc(test)
        scores.append((auc_score_train, auc_score))
        fprs.append(fpr)
        tprs.append(tpr)
    mean_auc = plot_roc_curve(fprs, tprs)
    print("mean auc", mean_auc)
    fprs.clear()
    tprs.clear()
    scores.clear()
    meanAucDf.loc[count] = [len(features), mean_auc]
    count = count + 1
    print("dropping", index)
    features = features.drop(index)

ax = plt.gca()

meanAucDf.plot(kind='line',x='numbers of features',y='mean auc',ax=ax)
plt.title(predict)
plt.xticks(np.arange(0, max_x_scale, step = 5))
plt.minorticks_on()
plt.savefig('meanAuc_vs_features.png', bbox_inches='tight')
plt.savefig('meanAuc_vs_features.pdf', bbox_inches='tight')
plt.show()




print("===================================================")















