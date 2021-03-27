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
from sklearn.metrics import roc_curve, auc
# FIXME: Matplotlib är version 3.2.2, du har version 3.2.1
import matplotlib.pyplot as plt
from IPython.display import display, HTML
# FIXME: IPython module ör version 7.16.1, du har version 7.15.0
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import os
import statistics_feature_importance
from sklearn.model_selection import StratifiedShuffleSplit
import Plot_ROC_AUC
import math


print(os.getcwd())

# NOTE: Ändrat så att man hämtar filen i arbetskatalogen då slipper du skriva path
# path = "/Users/Adam/PycharmProjects/Sommarforskning_test/Sepsis_test/"
path = os.getcwd()
item = "/Population_full.csv"

df = pd.read_csv(path+item, ';')
display(df.head(5))

print("Number of rows: ", df.shape[0])
counts = df.describe(include = 'all').iloc[0]
display(pd.DataFrame(counts.tolist(), columns=["Count of values"], index=counts.index.values).transpose())

features = df.drop(["ID", "Ålder", "Daysinadmission", "Död", "Daystodeath", "MortalityInhospital", "Mortality1day", "Mortality7days", "Mortality30days", "Mortality1Year", "Survival7days", "Prio", "Kön", "severe_sepsis"], axis=1).columns
predict = 'Mortality30days'

clf = BalancedRandomForestClassifier(n_estimators=100)

X = df.loc[:, features]
y = df[predict]

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

# ------- Create empty lists ----------
sensitivity = []
specificity = []
ppv = []
npv = []
LR_pos = []
LR_neg = []

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
    # Calculate confusion matrix
    predictions = clf.predict(X.iloc[test])
    cm = confusion_matrix(y.iloc[test], predictions)
    print(cm)
    # Calculate sensitivity
    sensitivity_cm = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    sensitivity.append(sensitivity_cm)
    # Calculate specificity
    specificity_cm = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity.append(specificity_cm)
    # Calculate ppv
    ppv_cm = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    ppv.append(ppv_cm)
    # Calculate npv
    npv_cm = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    npv.append(npv_cm)
    # Calculate LR+
    LR_pos_cm = sensitivity_cm / (1 - specificity_cm)
    LR_pos.append(LR_pos_cm)
    # Calculate LR-
    LR_neg_cm = (1 - sensitivity_cm) / specificity_cm
    LR_neg.append(LR_neg_cm)

    # ------------ Code below plots the confusion matrix ----------------------------------
    # Confusion_Matrix.plot_confusion_matrix(cm, classes=['Alive', 'Dead'], title='7-day mortality Confusion Matrix')
    # ------------ Code below show the 10 variables with highest feature importance --------
    # statistics_feature_importance.showStatistic(clf.feature_importances_, features)

    #----- Create DataFrame table with feature importance from all folders -----------------
    dFrame['importance_'+ str(index)] = clf.feature_importances_
    dFrame.index = features
    index = index + 1
    #---------------------------------------------------------------------------------------

print('------------------------------------------------------')

print('Accuracy', predict)
print()
#---------------Calculate mean, std and CI for sens, spec, etc..
Sample_size = math.sqrt(10)
print('Sample_size :', Sample_size)

mean_sensitivity = sum(sensitivity) / len(sensitivity)
print('mean sensitivity :', mean_sensitivity)
print('std sensitivity :', np.std(sensitivity))
CI_upper_sensitivity = mean_sensitivity + (1.96*np.std(sensitivity)/Sample_size)
CI_lower_sensitivity = mean_sensitivity - (1.96*np.std(sensitivity)/Sample_size)
print('CI upper sensitivity :', CI_upper_sensitivity)
print('CI lower sensitivity :', CI_lower_sensitivity)
print()

mean_specificity = sum(specificity) / len(specificity)
print('mean specificity :', mean_specificity)
print('std specificity :', np.std(specificity))
CI_upper_specificity = mean_specificity + (1.96*np.std(specificity)/Sample_size)
CI_lower_specificity = mean_specificity - (1.96*np.std(specificity)/Sample_size)
print('CI upper specificity :', CI_upper_specificity)
print('CI lower specificity :', CI_lower_specificity)
print()

mean_ppv = sum(ppv) / len(ppv)
print('mean ppv :', mean_ppv)
print('std ppv :', np.std(ppv))
CI_upper_ppv = mean_ppv + (1.96*np.std(ppv)/Sample_size)
CI_lower_ppv = mean_ppv - (1.96*np.std(ppv)/Sample_size)
print('CI upper ppv :', CI_upper_ppv)
print('CI lower ppv :', CI_lower_ppv)
print()

mean_npv = sum(npv) / len(npv)
print('mean npv :', mean_npv)
print('std npv :', np.std(npv))
CI_upper_npv = mean_npv + (1.96*np.std(npv)/Sample_size)
CI_lower_npv = mean_npv - (1.96*np.std(npv)/Sample_size)
print('CI upper npv :', CI_upper_npv)
print('CI lower npv :', CI_lower_npv)
print()

mean_LR_pos = sum(LR_pos) / len(LR_pos)
print('mean LR+ :', mean_LR_pos)
print('std LR+ :', np.std(LR_pos))
CI_upper_LR_pos = mean_LR_pos + (1.96*np.std(LR_pos)/Sample_size)
CI_lower_LR_pos = mean_LR_pos - (1.96*np.std(LR_pos)/Sample_size)
print('CI upper LR+ :', CI_upper_LR_pos)
print('CI lower LR+ :', CI_lower_LR_pos)
print()

mean_LR_neg = sum(LR_neg) / len(LR_neg)
print('mean LR- :', mean_LR_neg)
print('std LR- :', np.std(LR_neg))
CI_upper_LR_neg = mean_LR_neg + (1.96*np.std(LR_neg)/Sample_size)
CI_lower_LR_neg = mean_LR_neg - (1.96*np.std(LR_neg)/Sample_size)
print('CI upper LR- :', CI_upper_LR_neg)
print('CI lower LR- :', CI_lower_LR_neg)
print()

#------------- Calculate meanvalue for feature importance ------------------------------
featureMeanValueList = dFrame.mean(axis=1)
dFrame['meanValue'] = featureMeanValueList

#------------- index column name -------------------------------------------------------
dFrame.index.name = 'variables'

#------------- sort ascending for mean value -------------------------------------------
dFrameSorted = dFrame.sort_values(by='meanValue', ascending=True)

#------------ Export meanValue data table to Excel -------------------------------------
writer = pd.ExcelWriter('meanValueFolderData.xlsx', engine='xlsxwriter')

#--- Convert the meanvalue dataframe to an XlsxWriter Excel object ---------------------
dFrameSorted.to_excel(writer, sheet_name=predict)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
#---------------------------------------------------------------------------------------

# ----------- Plot ROC curve showing the auc for each fold + mean auc and std ----------
Plot_ROC_AUC.plot_roc_curve_Each_Fold(fprs, tprs, predict)


# ----------- Plot Roc curve showing only mean auc + std -------------------------------
Plot_ROC_AUC.plot_roc_curve_only_mean(fprs, tprs, predict)



print("===================================================")



