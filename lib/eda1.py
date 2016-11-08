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

def pause():
    os.system('read -p "Press Enter to continue..."')

def duration():
    global start
    
    end = time.clock()
    print '\nDuration: %.2f ' % (end - start)
    
    start = time.clock()

#load Dataframe from file/url
def loadDataframe(filename):
    print 'loading ' + filename + ' ...'
    return pandas.read_csv(filename, header=0, sep=';')

#drop not interesting columns and fill NaN values
def dataCleansing(dataframe):
    #axis: 0 for rows and 1 for columns
    dataframe.drop('cep', axis=1, inplace=True)
    dataframe.drop('sit_matricula', axis=1, inplace=True)

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
    if (createImages):
        print("histograms")
        dataframe.hist()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-histograms.png')
        imageidx += 1

    # density
    if (createImages):
        print("density")
        dataframe.plot(kind='density', subplots=True, sharex=False, legend=False)
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-density.png')
        imageidx += 1

    # box and whisker plots
    if (createImages):
        print("box and whisker plots")
        dataframe.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
        #dataframe.plot(kind='box', subplots=True, sharex=False, sharey=False)
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-box.png')
        imageidx += 1

    # scatter plot matrix
    if (createImages):
        print("scatter plot matrix")
        scatter_matrix(dataframe)
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-scatter-plot.png')
        imageidx += 1

    # correlation matrix
    if (createImages):
        print("correlation matrix")
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=True)
        cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-correlation-matrix.png')
        imageidx += 1

    # histograms of standardized data
    if (createImages):
        print("histograms of standardized data")
        array = dataframe.values
        Ax = array[:,0:ncolumns-1].astype(float)
        Ay = array[:,ncolumns-1]
        scaler = StandardScaler().fit(Ax)
        rescaledX = scaler.transform(Ax)
        stdDataframe = DataFrame(data=rescaledX)
        stdDataframe.hist()
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-standardized-histograms.png')
        imageidx += 1
    
    # density of standardized data
    if (createImages):
        print("density of standardized data")
        stdDataframe.plot(kind='density', subplots=True, sharex=False, legend=False)
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-standardized-density.png')
        imageidx += 1
    
    
    # box and whisker plots of standardized data
    if (createImages):
        print("box and whisker plots of standardized data")
        stdDataframe.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
        #stdDataframe.plot(kind='box', subplots=True, sharex=False, sharey=False)
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-standardized-box.png')
        imageidx += 1
    
    plt.close('all')

# Split-out validation dataset
def splitoutValidationDataset(dataframe):    
    print '\n=== Split-out train/validation datasets ==='

    ncolumns = dataframe.shape[1]
    array = dataframe.values
    
    X = array[:,0:ncolumns-1].astype(float)
    Y = array[:,ncolumns-1]

    X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=VALIDATION_SIZE, random_state=RAND_SEED)

    return (X_train, X_validation, Y_train, Y_validation)
    
#Feature Selection
def featureSelection(features, X_train, Y_train):
    print("\n=== Feature Selection ===")
    
    printFeaturesByRelevance(features, X_train, Y_train, ExtraTreesClassifier())
    printFeaturesByRelevance(features, X_train, Y_train, RandomForestClassifier())
    
    
#Print Features by Relevance
def printFeaturesByRelevance(features, X_train, Y_train, model):
    
    print "\nFeatures by Relevance (using '%s'):" % type(model).__name__
    model.fit(X_train, Y_train)

    idx = 0
    features_by_relevance = []
    for relevance in model.feature_importances_:
        features_by_relevance.append((relevance, features[idx]))
        idx += 1
    
    features_by_relevance = sorted(features_by_relevance, key=lambda x: x[0], reverse=True)

    for relevance, feature in features_by_relevance:
        print "%f : %s" % (relevance, feature)


# Evaluate Algorithms
def evaluteAlgorithms(X_train, Y_train, outputPath):
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
        msg = "%s:\tmean=%f (std=%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    if (createImages):
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-compare-algorithms.png')
        imageidx += 1

    plt.close('all')
    
    
# Standardize the dataset and reevaluate algorithms
def standardizeDataAndReevaluateAlgorithms(X_train, Y_train, outputPath):
    global imageidx
    print '\n === Standardize the dataset and reevaluate algorithms ==='
    
    
    #('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),
    
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('LR', LogisticRegression())])))
    pipelines.append(('ScaledLDA', Pipeline([('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
    pipelines.append(('ScaledKNN', Pipeline([('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
    pipelines.append(('ScaledCART', Pipeline([('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
    pipelines.append(('ScaledNB', Pipeline([('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('NB', GaussianNB())])))
    pipelines.append(('ScaledSVM', Pipeline([('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),('Scaler', StandardScaler()),('SVM', SVC())])))
    
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
    ensembles.append(('AB', AdaBoostClassifier()))
    ensembles.append(('GBM', GradientBoostingClassifier()))
    ensembles.append(('RF', RandomForestClassifier()))
    ensembles.append(('ET', ExtraTreesClassifier()))
    
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
    
    
def reset_imageidx(value=1):
    global imageidx
    imageidx = value
    
def set_createImages(value):
    global createImages
    createImages = value
    
    
    
def compareFeatureReductionTechniques(X_train, Y_train, outputPath):
    global imageidx

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', SVC())
    ])

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

    grid = GridSearchCV(pipe, cv=3, n_jobs=2, param_grid=param_grid)
    grid.fit(X_train, Y_train)
    
    mean_scores = numpy.array(grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (numpy.arange(len(N_FEATURES_OPTIONS)) *
                   (len(reducer_labels) + 1) + .5)

    if (createImages):
        plt.figure()
        COLORS = 'bgrcmyk'
        for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
            plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

        plt.title("Comparing feature reduction techniques")
        plt.xlabel('Reduced number of features')
        plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
        plt.ylabel('Digit classification accuracy')
        plt.ylim((0, 1))
        plt.legend(loc='upper left')
        #plt.show()
        plt.savefig(outputPath + str(imageidx).zfill(N_DIGITS) + '-Comparing-feature-reduction-techniques.png')
        imageidx += 1
    
    
# ===================================================
# ================== main function ==================
# ===================================================
def run(inputFilePath, outputPath, createImagesFlag):
    global imageidx
    global start

    print '####################################################################'
    print '############### Running Exploratory Data Analysis #1 ###############'
    print '####################################################################'
    print ''

    imageidx = 1
    start = time.clock()
    set_createImages(createImagesFlag)
    
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)    
        
    # Load dataset
    dataframe = loadDataframe(inputFilePath)
    dataframe = dataCleansing(dataframe)
    
    # Understand the data
    descriptiveStatistics(dataframe, outputPath)
    dataVisualizations(dataframe, outputPath)
        
    #Split-out train/validation dataset
    X_train, X_validation, Y_train, Y_validation = splitoutValidationDataset(dataframe)

    #Compare different feature reduction techniques
    compareFeatureReductionTechniques(X_train, Y_train, outputPath)
    
    # Select the most effective features
    featureSelection(dataframe.columns, X_train, Y_train)    
    
    # Evaluate Algorithms
    evaluteAlgorithms(X_train, Y_train, outputPath)
    
    # Standardize the dataset and reevaluate the same algorithms
    standardizeDataAndReevaluateAlgorithms(X_train, Y_train, outputPath)
    
    # Evaluate Ensemble Algorithms
    evaluateEnsembleAlgorith(X_train, Y_train, outputPath)  
    
    duration()    
    print '<<< THEN END - Running Exploratory Data Analysis #1 >>>'
    
