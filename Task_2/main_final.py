import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_validate
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import sklearn

import catboost as cb
from sklearn.metrics import mean_squared_error

from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

def get_last(X):
    return X.iloc[-1]

def get_score(df_true, df_submission):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1 = np.mean([sklearn.metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = sklearn.metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, sklearn.metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    return score

def main():
    # load data
    print("Loading Data")
    train_ft = pd.read_csv('handout/train_features.csv', index_col = 'pid')
    train_labels = pd.read_csv('handout/train_labels.csv', index_col = 'pid')
    test_ft = pd.read_csv('handout/test_features.csv', index_col = 'pid')

    # define the features for each subtask
    TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
            'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
            'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
    SEPSIS = ['LABEL_Sepsis']
    VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    print("Preprocessing Data")
    # ensure all times are in the correct window
    train_ft = train_ft.copy(deep=True)
    train_ft['Time'] = np.tile(np.arange(1,13),18995)
    test_ft = test_ft.copy(deep=True)
    test_ft['Time'] = np.tile(np.arange(1,13),12664)

    # remove outliers
    train_ft = train_ft.apply(lambda col: expit((col - np.mean(col))/np.std(col)))
    test_ft = test_ft.apply(lambda col: expit((col - np.mean(col))/np.std(col)))
    print("Imputing Data")
    # impute missing values
    imputer = IterativeImputer(max_iter=40, random_state=0,sample_posterior=False)


    train_ft.iloc[:, :] = imputer.fit_transform(train_ft)
    test_ft.iloc[:, :] = imputer.fit_transform(test_ft)
    print(train_ft.shape)
    print(train_ft.shape)
    print("Aggregating Data")
    # aggregate values for each patient
    train_ft_agg = train_ft.groupby(train_ft.index).agg(['mean','min', 'max','std', get_last])
    test_ft_agg = test_ft.groupby(test_ft.index).agg(['mean','min', 'max','std', get_last])

    # drop unnecessary columns
    train_ft_agg = train_ft_agg.drop(['Time',('Age', 'min'),('Age', 'std'),('Age', 'max'),('Age', 'get_last')], axis = 1)
    test_ft_agg = test_ft_agg.drop(['Time',('Age', 'min'),('Age', 'std'),('Age', 'max'),('Age', 'get_last')], axis = 1)

    # calculate correlations and drop columns with high correlation
    corr_matrix = train_ft_agg.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print(to_drop)

    train_ft_agg.drop(train_ft_agg[to_drop], axis=1,inplace=True)
    test_ft_agg.drop(test_ft_agg[to_drop], axis=1,inplace=True)


    # split into train and test
    X_train, X_val, y_train, y_val = train_test_split(train_ft_agg.sort_values('pid'), train_labels.sort_values('pid'), test_size = 0.2, random_state=5)
    X_test = test_ft_agg

    # sort the values by pid
    train_ft_agg.sort_values('pid',inplace=True)
    train_labels.sort_values('pid', inplace=True)
    X_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    X_val.sort_index(inplace=True)
    y_train.sort_index(inplace=True)
    y_val.sort_index(inplace=True)

    #  create test label dataframes
    y_test = pd.DataFrame(np.zeros((np.shape(test_ft_agg)[0], len(train_labels.columns.values))), columns = train_labels.columns.values)
    train_cv = y_val.copy()

    print("Looking for tests")
    for i in TESTS:
        print(i)
        X = train_ft_agg # X_train
        y = train_labels[i] #y_train[i]


        X, y = RandomUnderSampler(random_state=0).fit_resample(X, y)
        xgb_model = xgb.XGBClassifier()
        
        parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                'objective':['binary:logistic'],
                'learning_rate': [0.05], #so called `eta` value
                'max_depth': [4,5,6],
                'min_child_weight': [11],
                'subsample': [0.8],
                'colsample_bytree': [0.7],
                'n_estimators': [100, 200, 500], #number of trees, change it to 1000 for better results
                'missing':[-999],
                'verbosity':[0],
                'seed': [1337]}

        clf = GridSearchCV(xgb_model, parameters, n_jobs=1, 
                    cv=5, 
                    scoring='roc_auc',
                    verbose=2, refit=True)
        clf.fit(X, y)
        y_test.loc[:,i] = clf.predict_proba(X_test)[:,1]
        train_cv.loc[:,i] = clf.predict_proba(X_val)[:,1]

    print("SEPSIS TIME")
    for i in SEPSIS:
        print(i)
        X = train_ft_agg #X_train
        y = train_labels[i] #y_train[i]
        X, y = RandomUnderSampler(random_state=0).fit_resample(X, y)
        
        xgb_model = xgb.XGBClassifier()
        
        parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                'objective':['binary:logistic'],
                'learning_rate': [0.05, 0.1, 0.15, 0.5], #so called `eta` value
                'max_depth': [4, 5, 6, 7],
                'min_child_weight': [11],
                'subsample': [0.8],
                'colsample_bytree': [0.7],
                'n_estimators': [100, 200, 500, 1000], #number of trees, change it to 1000 for better results
                'missing':[-999],
                'verbosity':[0],
                'seed': [1337]}

        clf = GridSearchCV(xgb_model, parameters, n_jobs=1, 
                    cv=5, 
                    scoring='roc_auc',
                    verbose=2, refit=True)
        clf.fit(X, y)
        y_test.loc[:,i] = clf.predict_proba(X_test)[:,1]
        train_cv.loc[:,i] = clf.predict_proba(X_val)[:,1]
        

    pred = pd.DataFrame(columns=VITALS)

    print("VITALITY, K.O.")
    for label in VITALS:
        train_dataset = cb.Pool(X_train, y_train[label]) 

        model = cb.CatBoostRegressor(loss_function='RMSE')

        grid = {'iterations': [200, 400, 600],
            'learning_rate': [0.1, 0.5, 1.0],
            'depth': [4, 5, 6],
            'l2_leaf_reg': [0.5]}
        model.grid_search(grid, train_dataset)

        train_cv[label] = model.predict(X_val)
        y_test[label] = model.predict(X_test)
        

    # evaluate the performance
    print(get_score(y_val, train_cv))
    # save the predictions
    y_test.to_csv('prediction_final.zip', index=False, float_format='%.3f', compression='zip')

if __name__ == '__main__':
    main()