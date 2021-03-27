import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
import os
from sklearn.model_selection import StratifiedShuffleSplit

# Iteration of models developed using 91 features - 1 in each loop. Plot curve: AUC vs numbers of features.

path = os.getcwd()
item = "/Population_full.csv"

df = pd.read_csv(path + item, ';')

item_features = "/meanValueFolderData.xlsx"

meanValueDf = pd.read_excel(path + item_features)

list_of_features = meanValueDf['variables'].to_list()

features = df.drop(["ID", "Ålder", "Daysinadmission", "Död", "Daystodeath", "MortalityInhospital", "Mortality1day",
                    "Mortality7days", "Mortality30days", "Mortality1Year", "Survival7days", "Prio", "Kön",
                    "severe_sepsis"], axis=1).columns
predict = 'Mortality30days'

clf = BalancedRandomForestClassifier(n_estimators=100)


def calculate_roc_curve(fprs, tprs):
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Calculate ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Calculate the mean AUC and std AUC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return [mean_auc, std_auc]


def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8, random_state=0)

results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []

meanAucDf = pd.DataFrame(columns=['numbers of features', 'mean auc', 'std_auc'])

max_x_scale = len(features)
count = 0
for index in list_of_features:
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
    result_auc = calculate_roc_curve(fprs, tprs)
    print("mean auc :", result_auc[0])
    fprs.clear()
    tprs.clear()
    scores.clear()
    meanAucDf.loc[count] = [len(features), result_auc[0], result_auc[1]]
    count = count + 1
    print("dropping", index)
    features = features.drop(index)

plt.rcParams["figure.figsize"] = (20, 4.8)
ax = plt.gca()
tprs_upper = meanAucDf['mean auc'].to_numpy() + meanAucDf['std_auc'].to_numpy()
tprs_lower = meanAucDf['mean auc'].to_numpy() - meanAucDf['std_auc'].to_numpy()
A = meanAucDf['numbers of features'].to_numpy()
ax.fill_between(A, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

meanAucDf.plot(kind='line', x='numbers of features', y='mean auc', ax=ax)
plt.title(predict)
plt.xticks(np.arange(0, max_x_scale, step=5))
plt.yticks(np.arange(5, step=0.10))
plt.ylim(0.5, 0.9)
plt.xlim(1, 91)
plt.minorticks_on()
plt.savefig('meanAuc_vs_features.png', bbox_inches='tight')
plt.savefig('meanAuc_vs_features.pdf', bbox_inches='tight')
plt.show()
