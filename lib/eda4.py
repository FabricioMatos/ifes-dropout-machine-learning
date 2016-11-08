# License: BSD 3 clause <https://opensource.org/licenses/BSD-3-Clause>
# Copyright (c) 2016, Fabricio Vargas Matos <fabriciovargasmatos@gmail.com>
# All rights reserved.

''''
Delete columns not available at the end of first semester and run the same analysis of eda1.
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
import lib.eda2 as eda2


#constants
N_DIGITS = 3
NUM_FOLDS = 10
RAND_SEED = 7
SCORING = 'accuracy'
VALIDATION_SIZE = 0.20

#global variables
start = time.clock()
imageidx = 1
createImages = True

def duration():
    global start
    
    end = time.clock()
    print '\nDuration: %.2f ' % (end - start)
    
    start = time.clock()

        
# Standardize the dataset, reduce features using PCA and reevaluate the same algorithms
def standardizeDataAndReevaluateAlgorithms(X_train, Y_train, outputPath):
    global imageidx
    print '\n === Standardize the dataset and reevaluate algorithms ==='
    
    
    #('PCA', PCA()),
    
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('LR', LogisticRegression())])))
    pipelines.append(('ScaledLDA', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append(('ScaledKNN', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
    pipelines.append(('ScaledCART', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
    pipelines.append(('ScaledNB', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('NB', GaussianNB())])))
    pipelines.append(('ScaledSVM', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('SVM', SVC())])))
    
    results = []
    names = []
    
    for name, model in pipelines:
        kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=SCORING)
        results.append(cv_results)
        names.append(name)
        msg = "%s:\tmean=%f (std=%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    # Compare Algorithms
    if (createImages):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison - Standardized Dataset')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-compare-algorithms-standardized-dataset.png')
        imageidx += 1

    plt.close('all')
    
    
# Evaluate Ensemble Algorithms
def evaluateEnsembleAlgorith(X_train, Y_train, outputPath):
    global imageidx
    print '\n === Evaluate Ensemble Algorithms ==='

    ensembles = []
    ensembles.append(('AB', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('AB', AdaBoostClassifier())])))
    ensembles.append(('GBM', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier())])))
    ensembles.append(('RF', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))
    ensembles.append(('ET', Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))
    
    results = []
    names = []
    
    for name, model in ensembles:
        kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=SCORING)
        results.append(cv_results)
        names.append(name)
        msg = "%s:\tmean=%f (std=%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    if (createImages):
        fig = plt.figure()
        fig.suptitle('Ensemble Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-Ensemble-Algorithm-Compariso.png')
        imageidx += 1
    
    plt.close('all')
    
    
# Tune scaled SVM
def tuneSVM(X_train, Y_train, outputPath):
    print 'tune SVM'

    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

    c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = dict(C=c_values, kernel=kernel_values)
    
    model = SVC()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)    
    for params, mean_score, scores in grid_scores:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        
        
# ===================================================
# ================== main function ==================
# ===================================================
def run(inputFilePath, outputPath, createImagesFlag):
    global start

    print '####################################################################'
    print '############### Running Exploratory Data Analysis #3 ###############'
    print '####################################################################'
    print ''
    
    start = time.clock()
    eda1.reset_imageidx()
    eda1.set_createImages(createImagesFlag)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)    
        
    # Load dataset
    dataframe = eda1.loadDataframe(inputFilePath)
    
    # drop out 'not fair' features
    dataframe = eda2.dataCleansing(dataframe)
    dataframe.to_csv(inputFilePath[:-4] + '-cleaned.csv')
    
        
    # Understand the data
    #eda1.descriptiveStatistics(dataframe, outputPath)
    #eda1.dataVisualizations(dataframe, outputPath)
        
    #Split-out train/validation dataset
    X_train, X_validation, Y_train, Y_validation = eda1.splitoutValidationDataset(dataframe)    

    '''
    ScaledLR:	mean=0.816812 (std=0.050291)
    ScaledLDA:	mean=0.812367 (std=0.038609)
    ScaledSVM:	mean=0.812319 (std=0.047945)    
    ScaledNB:	mean=0.790338 (std=0.044079)
    ScaledKNN:	mean=0.772512 (std=0.050124)
    ScaledCART:	mean=0.739614 (std=0.043244)
    '''
    
    tuneSVM(X_train, Y_train, outputPath)
    
    
    
    duration()    
    print '<<< THEN END - Running Exploratory Data Analysis #3 >>>'
    
