# -------- Virtual Enviroment added modules ---------
# pandas module
# sklearn module
# matplotlib module
# IPython module
# Imblearn module
# xlsxwriter module
# ---------------------------------------------------

import pandas as pd
# FIXME: Pandas module är version 1.05, du har ver 1.04
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib
# FIXME: Matplotlib är version 3.2.2, du har version 3.2.1
import matplotlib.pyplot as plt
from IPython.display import display, HTML
# FIXME: IPython module ör version 7.16.1, du har version 7.15.0
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import itertools
from sklearn.tree import export_graphviz
from subprocess import call
#from IPython.display import Image
import os
import statistics


print(os.getcwd())

# NOTE: Ändrat så att man hämtar filen i arbetskatalogen då slipper du skriva path
# path = "/Users/Adam/PycharmProjects/Sommarforskning_test/Sepsis_test/"
path = os.getcwd()
item = "\Population_full.csv"

df = pd.read_csv(path+item,';')
display(df.head(5))

print("Number of rows: ", df.shape[0])
counts = df.describe().iloc[0]
display(
    pd.DataFrame(
        counts.tolist(),
        columns=["Count of values"],
        index=counts.index.values
    ).transpose()
)

 # df = df.drop(["Non_Survivors", "severe_sepsis], axis=1)
features = df.drop(["ID", "Ålder", "Daysinadmission", "Död", "Daystodeath", "MortalityInhospital", "Mortality1day", "Mortality7days", "Mortality30days", "Mortality1Year", "Survival7days", "Prio", "Kön", "severe_sepsis"], axis=1).columns # Adam
predict = 'Mortality7days'

# A = df['Mortality7days']
# type(A)



clf = BalancedRandomForestClassifier(n_estimators=100)

X = df.loc[:, features]
y = df[predict]

def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates."""

    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14, 10))

    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    # TODO: Stopp Blocking mode, Grafen stoppar koden tills fönstret stängs
    plt.show()
    return (f, ax)


def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:, 1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

print("++++++++++++++++++++++++++++++++++++++++++++++++++")

cv = StratifiedKFold(n_splits=10, random_state=123, shuffle=True)
results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []

# Skapa objekt för medelvärde för beräkning
dFrame = pd.DataFrame()
index = 0
for (train, test), i in zip(cv.split(X, y), range(10)):
    clf.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)
    # Debug Print Importance + Feature lista (10st med högsta värdena)
    statistics.showStatistic(clf.feature_importances_, features)

    #-------- Create DataFrame table with all folders ----------------
    dFrame['importance_'+ str(index)] = clf.feature_importances_
    dFrame.index = features
    index = index + 1
    #------------------------------------------------------------------

# Calculate meanvalue for each row
featureMeanValueList = dFrame.mean(axis=1)
dFrame['meanValue'] = featureMeanValueList

#------------ Export meanValue data table to Excel ---------------------
writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
dFrame.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
#------------------------------------------------------------------------

plot_roc_curve(fprs, tprs);

pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])

print("===================================================")

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111)

print("SISTA UTSKRIFTEN")
# Debug Print Importance + Feature lista (10st med högsta värdena)
statistics.showStatistic(clf.feature_importances_, features)

# df_f = pd.DataFrame(clf.feature_importances_, columns=["importance"])
# df_f["labels"] = features
# df_f.sort_values("importance", inplace=True, ascending=False)
# display(df_f.head(10))

index = np.arange(len(clf.feature_importances_))
bar_width = 0.5
rects = plt.barh(index, df_f["importance"], bar_width, alpha=0.4, color='b', label='Main')
plt.yticks(index, df_f["labels"])
# plt.show()

# FIXME: Probe är okänd
# df_test["prob_true"] = probs[:, 1]
# df_risky = df_test[df_test["prob_true"] > 0.9]
# display(df_risky.head(5)[["prob_true"]])

