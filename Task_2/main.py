import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')
test_features = pd.read_csv('test_features.csv')


X = np.array(train_features)
y_sub1 = np.array(train_labels[['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 
    'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 
    'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']])
y_sub2 = np.array(train_labels[['pid', 'LABEL_Sepsis']])
y_sub3 = np.array(train_labels[['pid', 'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']])
X_test = np.array(test_features)

#reorder X
j = 0
l = len(X[0])
reordX = np.empty((len(X)//12, l - 1))
for i in range(0, len(X), 12):
    #age
    reordX[j, 0] = X[i, 2]
    for feature in range(0, l - 3):
        avg = 0
        nnan = 0
        for time in range(0, 12):
            if not np.isnan(X[i + time, 3 + feature]):
                avg += X[i + time, 3 + feature]
                nnan += 1
        if nnan == 0:
            nnan = 1
        reordX[j, 1 + feature] = avg/nnan
    j += 1
#print(reordX)

#reorder X_test
j = 0
l = len(X_test[0])
pid = np.empty((len(X_test)//12))
reordX_test = np.empty((len(X_test)//12, l - 1))
for i in range(0, len(X_test), 12):
    #pid
    pid[j] = X_test[i, 0]
    #age
    reordX_test[j, 0] = X_test[i, 2]
    for feature in range(0, l - 3):
        avg = 0
        nnan = 0
        if nnan == 0:
            nnan = 1
        for time in range(0, 12): 
            if not np.isnan(X_test[i + time, 3 + feature]):
                avg += X_test[i + time, 3 + feature]
                nnan += 1
        reordX_test[j, 1 + feature] = avg/nnan
    j += 1
#print(reordX_test)

#sub1
sub1 = np.empty(len(reordX_test))
for i in range(10):
    log_reg1 = LogisticRegression(solver='liblinear', max_iter=200).fit(reordX, y_sub1[:, i + 1])
    out = log_reg1.predict_proba(reordX_test)[:, 1]
    #print(out)
    sub1 = np.c_[sub1, out]
sub1 = sub1[:, 1:]

#sub2
log_reg2 = LogisticRegression(solver='liblinear', max_iter=200).fit(reordX, y_sub2[:, 1])
sub2 = log_reg2.predict_proba(reordX_test)[:, 1]
#print(sub2)
sepsis_classifier = make_pipeline(StandardScaler(),BalancedRandomForestClassifier(n_estimators=500, random_state=0,n_jobs=-1))

#sub3
sub3 = np.empty(len(reordX_test))
for i in range(4):
    lin_reg3 = LinearRegression().fit(reordX, y_sub3[:, i + 1])
    out = lin_reg3.predict(reordX_test)
    #print(out)
    sub3 = np.c_[sub3, out]
sub3 = sub3[:, 1:]


#output
output = np.c_[pid, sub1, sub2, sub3]
#print(output)
output = pd.DataFrame(output)

output.columns = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 
                             'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2', 
                             'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis',
                             'LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

output.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')
