import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from pprint import pprint
import joblib
import statistics

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection._split import check_cv
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, cross_val_score, RandomizedSearchCV, cross_val_predict

def feature_selection(df, nonzero_thrd = 0.0, cor_thrd = 0.95):

    selector = VarianceThreshold(nonzero_thrd)
    selector.fit(df)
    nonzero_df = df[df.columns[selector.get_support(indices=True)]]
    corr_matrix = nonzero_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cor_thrd)]
    
    return nonzero_df.drop(nonzero_df[to_drop], axis=1)

def feature_norm_fit(train_df , scaler = MinMaxScaler()):
    array =  train_df.values
    df_norm = pd.DataFrame(scaler.fit_transform(array), columns=train_df.columns, index=train_df.index)
    return df_norm, scaler

def feature_norm_transform(test_df, scaler):
    array =  test_df.values
    df_norm = pd.DataFrame(scaler.transform(array), columns=test_df.columns, index=test_df.index)
    return df_norm  

def test_feature(df, feature, scaler = None):
    
    with open('../data/Descriptors/filtered_features.json') as f:
        dict_features = json.load(f)
        
    if feature not in dict_features.keys():
        raise Exception(f'The feature **{feature}** is not support, please choose from [ecfp6_bits, ecfp6_counts, maccs, rdkit2d, mordred]')
        
    filtered_desc = dict_features[feature]
    df = df[filtered_desc]
    
    if scaler:
        df = feature_norm_transform(df, scaler)
    
    return df



def prepare_input(df_label, df_feature, target, encoder = None):
    df_labled = df_label[~df_label[target].isnull()]
    df_unbeled = df_label[df_label[target].isnull()]
    labeled_feature = df_feature.loc[df_labled.index].values.astype('float32')
    ublabeled_feature = df_feature.loc[df_unbeled.index].values.astype('float32')
    labeled_Y = df_labled[target].values
    if encoder:
        labeled_Y = encoder.transform(labeled_Y)

    return labeled_feature, ublabeled_feature, labeled_Y, df_labled.index, df_unbeled.index

def model_selection(model, params_grid, X, y, 
                    scoring = None, cv=5, n_jobs=6, GridSearch = True, n_iter=20, refit = True):

    if GridSearch == True:
        model_train = GridSearchCV(model, params_grid, cv=cv, n_jobs=n_jobs, scoring = scoring,refit = refit)
    else:
        model_train = RandomizedSearchCV(model, param_distributions = params_grid, 
                                         n_iter = n_iter,cv=cv, n_jobs=n_jobs, scoring=scoring, refit = refit)
    
    model_train.fit(X, y)
    print("Best parameters set found on development set:", model_train.best_params_ )
    print("Best score:", model_train.best_score_ )
    
    print("Grid scores on development set:")
    print()
    means = model_train.cv_results_['mean_test_score']
    stds = model_train.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_train.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return model_train

def Regression_meta_features(model,X,y,NaN, 
                 index_X, index_NaN, col_names = ['RM_pred'], cv=10, n_jobs = 6):

    regression_scoring = {'RMSE': make_scorer(rmse), 'R2': 'r2', 
                      'MAE': make_scorer(mean_absolute_error),
                     'MSE': make_scorer(mean_squared_error)}
    
    np.random.seed(1234) # get reproducible results
    kfold = KFold(n_splits=cv, shuffle=True, random_state=15)       
    
    instance = clone(model)
    instance.fit(X, y)
        
    NaN_fold_predictions = instance.predict(NaN)        
    out_of_fold_predictions = cross_val_predict(model, X, y, cv=kfold, method = 'predict', n_jobs = n_jobs)
    
    cv_score = cross_validate(model, X, y, cv=kfold, n_jobs = n_jobs, scoring = regression_scoring)
    
    OOF_predictions = pd.DataFrame(out_of_fold_predictions, index = index_X, columns=col_names)     
    NaN_predictions = pd.DataFrame(NaN_fold_predictions, index = index_NaN, columns=col_names)
    meta_feature = pd.concat([OOF_predictions, NaN_predictions])
                         
    return meta_feature, out_of_fold_predictions, instance, cv_score


def rmse(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def report_reg_models(score):
    print('RMSE:', rmse(y_true, y_pred))
    print('R2:', r2_score(y_true, y_pred))
    print('MAE', mean_absolute_error(y_true, y_pred))
    print('MSE', mean_squared_error(y_true, y_pred))
    
def report_cv_reg_models(score, decimal = 3):
    print('RMSE:', round(statistics.mean(score['test_RMSE']), decimal), 'std:',
                   round(statistics.stdev(score['test_RMSE']), decimal))

    print('R2:', round(statistics.mean(score['test_R2']), decimal), 'std:',
                   round(statistics.stdev(score['test_R2']), decimal))
    
    print('MAE:', round(statistics.mean(score['test_MAE']), decimal), 'std:',
                   round(statistics.stdev(score['test_MAE']), decimal))
    
    print('MSE:', round(statistics.mean(score['test_MSE']), decimal), 'std:',
                   round(statistics.stdev(score['test_MSE']), decimal))
    
def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def Classification_meta_features(model,X,y,NaN, 
                 index_X, index_NaN, col_names, cv=10, Probs=True, n_jobs = 6):
    classification_scoring = {'Accuracy': make_scorer(accuracy_score), 
                              'Balance Accuracy': make_scorer(balanced_accuracy_score), 
                      'matthews_corrcoef': make_scorer(matthews_corrcoef),
                     'f1_score': make_scorer(f1_score, average='weighted'),
                     'AUROC': make_scorer(multiclass_roc_auc_score)    
                             }
    
    np.random.seed(1234) 
    kfold = KFold(n_splits=cv, shuffle=True, random_state=15)    
    
    instance = clone(model)
    instance.fit(X, y)
        
    if Probs == True:
        NaN_fold_predictions = instance.predict_proba(NaN)
        out_of_fold_predictions = cross_val_predict(model, X, y, cv=kfold, method = 'predict_proba', n_jobs = n_jobs)
        
        cv_score = cross_validate(model, X, y, cv=kfold, n_jobs = n_jobs, scoring = classification_scoring)
        
        OOF_predictions = pd.DataFrame(out_of_fold_predictions, index = index_X, columns=col_names)     
        NaN_predictions = pd.DataFrame(NaN_fold_predictions, index = index_NaN, columns=col_names)
    
        meta_feature = pd.concat([OOF_predictions, NaN_predictions])
            
    else: 
        NaN_fold_predictions = instance.predict(NaN)        
        out_of_fold_predictions = cross_val_predict(model, X, y, cv=kfold, method = 'predict', n_jobs = n_jobs)
        
        cv_score = cross_validate(model, X, y, cv=kfold, n_jobs = n_jobs, scoring = classification_scoring)
        
        OOF_predictions = pd.DataFrame(out_of_fold_predictions, index = index_X, columns=col_names)     
        NaN_predictions = pd.DataFrame(NaN_fold_predictions, index = index_NaN, columns=col_names)
        meta_feature = pd.concat([OOF_predictions, NaN_predictions])
        
    return meta_feature, out_of_fold_predictions, instance, cv_score

def report_clf_models(score, decimal = 3):
    print('Accuracy:', round(statistics.mean(score['test_Accuracy']), decimal), 'std:',
                   round(statistics.stdev(score['test_Accuracy']), decimal))

    print('Balance Accuracy:', round(statistics.mean(score['test_Balance Accuracy']), decimal), 'std:',
                   round(statistics.stdev(score['test_Balance Accuracy']), decimal))
    
    print('matthews_corrcoef:', round(statistics.mean(score['test_matthews_corrcoef']), decimal), 'std:',
                   round(statistics.stdev(score['test_matthews_corrcoef']), decimal))
    
    print('f1_score:', round(statistics.mean(score['test_f1_score']), decimal), 'std:',
                   round(statistics.stdev(score['test_f1_score']), decimal))

    print('AUROC:', round(statistics.mean(score['test_AUROC']), decimal), 'std:',
                   round(statistics.stdev(score['test_AUROC']), decimal))
    
def prob_to_pred(probs):
    classes = probs.argmax(axis=-1)
    return classes