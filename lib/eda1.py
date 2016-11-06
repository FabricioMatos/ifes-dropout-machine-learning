# License: BSD 3 clause <https://opensource.org/licenses/BSD-3-Clause>
# Copyright (c) 2016, Fabricio Vargas Matos <fabriciovargasmatos@gmail.com>
# All rights reserved.

''''
Run the first exploratory data analysis in order to underdand better the data available and plan the next steps.
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

#constants
N_DIGITS = 3
NUM_FOLDS = 10
RAND_SEED = 7
SCORING = 'accuracy'
VALIDATION_SIZE = 0.20


#global variables
start = time.clock()
imageidx = 1


def pause():
    os.system('read -p "Press Enter to continue..."')

def duration():
    global start
    
    end = time.clock()
    print 'Duration: %.2f ' % (end - start)
    
    start = time.clock()

#load Dataframe from file/url
def loadDataframe(filename):
    print 'loading ' + filename + ' ...'
    return pandas.read_csv(filename, header=0, sep=';')

#drop not interesting columns and fill NaN values
def dataCleansing(dataframe):
    #axis: 0 for rows and 1 for columns
    dataframe.drop('CEP', axis=1, inplace=True)
    dataframe.drop('SIT_MATRICULA', axis=1, inplace=True)

    #replace NaN with 0
    dataframe.fillna(value=0, inplace=True)
    
    return dataframe
    
    
# Descriptive statistics
def descriptiveStatistics(dataframe, outputPath):
    
    # Summarize Data
    print("\n=== Summarize Data ===")

    # shape
    print(dataframe.shape)

    # types
    pandas.set_option('display.max_rows', 500)
    print(dataframe.dtypes)

    # head
    pandas.set_option('display.width', 100)
    print(dataframe.head(20))

    # descriptions, change precision to 3 places
    pandas.set_option('precision', 3)
    print(dataframe.describe())

    # class distribution
    print(dataframe.groupby('evadiu').size())


# Data visualizations
def dataVisualizations(dataframe, outputPath):
    global imageidx
    
    ncolumns = dataframe.shape[1]
    
    print("\n=== Data visualizations ===")

    # histograms
    print("histograms")
    dataframe.hist()
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-histograms.png')
    imageidx += 1

    # density
    print("density")
    dataframe.plot(kind='density', subplots=True, layout=(6,7), sharex=False, legend=False)
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-density.png')
    imageidx += 1

    # box and whisker plots
    print("box and whisker plots")
    dataframe.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False)
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-box.png')
    imageidx += 1

    # scatter plot matrix
    print("scatter plot matrix")
    scatter_matrix(dataframe)
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-scatter-plot.png')
    imageidx += 1

    # correlation matrix
    print("correlation matrix")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-correlation-matrix.png')
    imageidx += 1


    # histograms of standardized data
    print("histograms of standardized data")
    array = dataframe.values
    Ax = array[:,0:ncolumns-1].astype(float)
    Ay = array[:,ncolumns-1]
    scaler = StandardScaler().fit(Ax)
    rescaledX = scaler.transform(Ax)
    df = DataFrame(data=rescaledX)
    df.hist()
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-standardized-histograms.png')
    imageidx += 1

# Split-out validation dataset
def splitoutValidationDataset(dataframe):    
    print '\n=== Split-out train/validation datasets ==='

    ncolumns = dataframe.shape[1]
    array = dataframe.values
    
    X = array[:,0:ncolumns-1].astype(float)
    Y = array[:,ncolumns-1]

    X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=VALIDATION_SIZE, random_state=RAND_SEED)

    return (X_train, X_validation, Y_train, Y_validation)
    


# Evaluate Algorithms
def evaluteAlgorithms(X_train, X_validation, Y_train, Y_validation, outputPath):
    global imageidx
    
    print '\n=== Evaluate algorithms ==='
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    results = []
    names = []


    for name, model in models:
        kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=SCORING)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-compare-algorithms.png')
    imageidx += 1

    
    
# Standardize the dataset and reevaluate algorithms
def standardizeDataAndReevaluateAlgorithms(X_train, X_validation, Y_train, Y_validation, outputPath):
    global imageidx

    print '\n === Standardize the dataset and reevaluate algorithms ==='
    
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
    pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
    pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
    pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
    results = []
    names = []
    
    for name, model in pipelines:
        kfold = cross_validation.KFold(n=len(X_train), n_folds=NUM_FOLDS, random_state=RAND_SEED)
        cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=SCORING)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison - Standardized Dataset')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-compare-algorithms-standardized-dataset.png')
    imageidx += 1

    
# ===================================================
# ================== main function ==================
# ===================================================
def run(inputPath='../input/', outputPath='../output/'):
    global imageidx

    print '<<< Running Exploratory Data Analysis #1 ==='

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)    
        
    # Load dataset
    dataframe = loadDataframe(inputPath + 'curso_1200.csv')
    dataframe = dataCleansing(dataframe)
    
    # Understand the data
    descriptiveStatistics(dataframe, outputPath)
    dataVisualizations(dataframe, outputPath)
        
    #Split-out train/validation dataset
    X_train, X_validation, Y_train, Y_validation = splitoutValidationDataset(dataframe)
    
    # Evaluate Algorithms
    evaluteAlgorithms(X_train, X_validation, Y_train, Y_validation, outputPath)
    
    # Standardize the dataset and reevaluate the same algorithms
    standardizeDataAndReevaluateAlgorithms(X_train, X_validation, Y_train, Y_validation, outputPath)
    
    print '\n'
    duration()
    
    print '=== Running Exploratory Data Analysis #1 >>>'
    
    
    

