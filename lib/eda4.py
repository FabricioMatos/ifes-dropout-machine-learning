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
results = []
names = []
params = []


def tuneLR(X_train, Y_train, outputPath):
    global results, names, params
    
    print 'tune LR'
    
    pipeline = Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler())])
    scaler = pipeline.fit(X_train)
    rescaledX = scaler.transform(X_train)
    
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = dict(C=c_values)
    
    model = LogisticRegression()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    best_idx = grid_result.best_index_

    #TODO: check it out if 'mean_test_score' is really what a want here
    cv_results = grid_result.cv_results_['mean_test_score']
    
    results.append(cv_results)
    names.append('LR')
    params.append('%s' % grid_result.cv_results_['params'][best_idx])
        
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)    
    for param, mean_score, scores in grid_scores:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), param))

def tuneLDA(X_train, Y_train, outputPath):
    global results, names, params
    
    print 'tune LDA'
    
    pipeline = Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler())])
    scaler = pipeline.fit(X_train)
    rescaledX = scaler.transform(X_train)
    
    #http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
    tol_values = [0.00001, 0.0001, 0.001, 0.01]
    solver_values = ['svd', 'lsqr', 'eigen']
    param_grid = dict(tol=tol_values, solver=solver_values)
    
    model = LinearDiscriminantAnalysis()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    best_idx = grid_result.best_index_

    #TODO: check it out if 'mean_test_score' is really what a want here
    cv_results = grid_result.cv_results_['mean_test_score'] 
    
    results.append(cv_results)
    names.append('LDA')
    params.append('%s' % grid_result.cv_results_['params'][best_idx])
        
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)    
    for param, mean_score, scores in grid_scores:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), param))
    
    
# Tune scaled SVM
def tuneSVM(X_train, Y_train, outputPath):
    global results, names, params
    
    print 'tune SVM'

    pipeline = Pipeline([('PCA', PCA()),('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler())])
    scaler = pipeline.fit(X_train)
    rescaledX = scaler.transform(X_train)
    
    c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = dict(C=c_values, kernel=kernel_values)
    
    model = SVC()
    
    kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=SCORING, cv=kfold)
    
    grid_result = grid.fit(rescaledX, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))    
        
    best_idx = grid_result.best_index_

    #TODO: check it out if 'mean_test_score' is really what a want here
    cv_results = grid_result.cv_results_['mean_test_score']
    
    results.append(cv_results)
    names.append('SVM')
    params.append('%s' % grid_result.cv_results_['params'][best_idx])
        
    grid_scores = sorted(grid_result.grid_scores_, key=lambda x: x[2].mean(), reverse=True)    
    for param, mean_score, scores in grid_scores:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), param))
        
        
def drawTunedAlgorithmsComparison(results, names, outputPath):
    global imageidx
    print '\n === Tuned Algorithms Comparison ===\n'

    algorithms = numpy.dstack((names,params, results))[0]    
    for name, param, result in algorithms:
        print "%f (%f) - %s => Best Params: %s" % (result.mean(), result.std(), name, param) 
    
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
        
# ===================================================
# ================== main function ==================
# ===================================================
def run(inputFilePath, outputPath, createImagesFlag):
    global start

    print '####################################################################'
    print '############### Running Exploratory Data Analysis #4 ###############'
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
            
    #Split-out train/validation dataset
    X_train, X_validation, Y_train, Y_validation = eda1.splitoutValidationDataset(dataframe)    

    # tune each algorithm
    tuneLR(X_train, Y_train, outputPath)
    tuneLDA(X_train, Y_train, outputPath)
    tuneSVM(X_train, Y_train, outputPath)
    
    #print the results comparing the algorithms with the best tune for each one
    drawTunedAlgorithmsComparison(results, names, outputPath)
    
    print '\n<<< THEN END - Running Exploratory Data Analysis #4 >>>'
    
    