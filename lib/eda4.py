# License: BSD 3 clause <https://opensource.org/licenses/BSD-3-Clause>
# Copyright (c) 2016, Fabricio Vargas Matos <fabriciovargasmatos@gmail.com>
# All rights reserved.

''''
Tune the 3 most promissing algorithms and compare them
'''

# Load libraries
import os
import time
import pandas
import numpy
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2

import lib.eda1 as eda1
import lib.eda3 as eda3


#constants
N_DIGITS = 3
NUM_FOLDS = 10
RAND_SEED = 7
SCORING = 'accuracy'
VALIDATION_SIZE = 0.20
N_JOBS = 6

#global variables
start = time.clock()
imageidx = 1
createImages = True
results = []
names = []
params = []
bestResults = []

# RandomForestClassifier
def tuneRF(X_train, Y_train, outputPath):
    global results, names, params, bestResults
    
    print 'tune LR (Random Forest Classifier)'
    
    pipeline = Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler())])
    scaler = pipeline.fit(X_train)
    rescaledX = scaler.transform(X_train)

    #tune para meters
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #n_estimators_values = [5, 10, 100, 1000, 3000]
    n_estimators_values = [1000]
    max_features_values = [0.1, 'auto', 'sqrt', 'log2', None] # (float)0.1=>10%
    criterion_values = ['gini', 'entropy']
    
    param_grid = dict(n_estimators=n_estimators_values, max_features=max_features_values, criterion=criterion_values)
    
    model = RandomForestClassifier()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(n_jobs=N_JOBS, verbose=10, estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    best_idx = grid_result.best_index_

    #TODO: check it out if 'mean_test_score' is really what I want here
    cv_results = grid_result.cv_results_['mean_test_score']
    results.append(cv_results)
    
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)
    first = True
    for param, mean_score, scores in grid_scores:
        if first:
            bestResults.append({'name':'RF', 'mean':scores.mean(), 'std':scores.std(), 'params':param})
            first = False
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), param))

# ExtraTreesClassifier
def tuneET(X_train, Y_train, outputPath):
    global results, names, params, bestResults
    
    print 'tune ET (Extra Trees Classifier)'
    
    pipeline = Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler())])
    scaler = pipeline.fit(X_train)
    rescaledX = scaler.transform(X_train)
    
    #tune para meters
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    #n_estimators_values = [5, 10, 100, 1000, 3000]
    n_estimators_values = [1000]
    max_features_values = [0.1, 'auto', 'sqrt', 'log2', None] # (float)0.1=>10%
    criterion_values = ['gini', 'entropy']
    
    param_grid = dict(n_estimators=n_estimators_values, max_features=max_features_values, criterion=criterion_values)
    
    model = ExtraTreesClassifier()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(n_jobs=N_JOBS, verbose=10, estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    best_idx = grid_result.best_index_

    #TODO: check it out if 'mean_test_score' is really what a want here
    cv_results = grid_result.cv_results_['mean_test_score']
    results.append(cv_results)
    
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)
    first = True
    for param, mean_score, scores in grid_scores:
        if first:
            bestResults.append({'name':'ET', 'mean':scores.mean(), 'std':scores.std(), 'params':param})
            first = False
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), param))
    
    
# Tune scaled SVM
def tuneSVM(X_train, Y_train, outputPath):
    global results, names, params, bestResults
    
    print 'tune SVM (Support Vector Machines Classifier)'

    pipeline = Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler())])
    scaler = pipeline.fit(X_train)
    rescaledX = scaler.transform(X_train)
    
    #c_values = [0.1, 1.0, 100.0, 10000.0, 100000.0]
    c_values = [10000.0, 100000.0]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = dict(C=c_values, kernel=kernel_values)
    
    model = SVC()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(n_jobs=N_JOBS, verbose=10, estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    best_idx = grid_result.best_index_

    #TODO: check it out if 'mean_test_score' is really what a want here
    cv_results = grid_result.cv_results_['mean_test_score']
    results.append(cv_results)
    
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)
    first = True
    for param, mean_score, scores in grid_scores:
        if first:
            bestResults.append({'name':'SVM', 'mean':scores.mean(), 'std':scores.std(), 'params':param})
            first = False
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), param))
        
        
def drawTunedAlgorithmsComparison(results, names, outputPath):
    global imageidx
    print '\n === Tuned Algorithms Comparison ===\n'

    #print bestResults
    for x in bestResults:
        print x
            
    # Compare Algorithms
    if (createImages):
        fig = plt.figure()
        fig.suptitle('Final Tuned-Algorithms Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-Tuned-Algorithm-Comparison.png')
        imageidx += 1
    
    plt.close('all')
        
        
def set_createImages(value):
    global createImages
    createImages = value
    
        
# ===================================================
# ================== main function ==================
# ===================================================
def run(inputFilePath, outputPath, createImagesFlag, dropColumns):
    global start

    print '####################################################################'
    print '############### Running Exploratory Data Analysis #4 ###############'
    print '####################################################################'
    print ''
    
    set_createImages(createImagesFlag)
    start = time.clock()
    eda1.reset_imageidx()
    eda1.set_createImages(createImagesFlag)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)    
        
    # Load dataset
    dataframe = eda1.loadDataframe(inputFilePath)
    
    # drop out 'not fair' features
    dataframe = eda1.dataCleansing(dataframe, dropColumns)
            
    #Split-out train/validation dataset
    X_train, X_validation, Y_train, Y_validation = eda1.splitoutValidationDataset(dataframe)    

    '''
    # tune each algorithm
    try:
        tuneRF(X_train, Y_train, outputPath)
    except Exception as e:
        print "ERROR: couldn't tune RF"
        print "Message: %s" % str(e)
        
    try:
        tuneET(X_train, Y_train, outputPath)
    except Exception as e:
        print "ERROR: couldn't tune ET"
        print "Message: %s" % str(e)
    '''  
        
    try:
        tuneSVM(X_train, Y_train, outputPath)
    except Exception as e:
        print "ERROR: couldn't tune SVM"
        print "Message: %s" % str(e)
    
    #print the results comparing the algorithms with the best tune for each one
    drawTunedAlgorithmsComparison(results, names, outputPath)
    
    print '\n<<< THEN END - Running Exploratory Data Analysis #4 >>>'
    
    #RF - Best: 0.853451 using {'max_features': 'log2', 'n_estimators': 1000, 'criterion': 'gini'}
    #ET - Best: 0.855320 using {'max_features': None, 'n_estimators': 1000, 'criterion': 'gini'}