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

start = time.clock()

def pause():
    os.system('read -p "Press Enter to continue..."')

def duration(start):
    end = time.clock()
    print 'Duration: %.2f ' % (end - start)

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
    print("=== Summarize Data ===")

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

    ncolumns = dataframe.shape[1]
    
    imageidx = 1
    ndigits = 3

    print("=== Data visualizations ===")

    # histograms
    print("histograms")
    dataframe.hist()
    plt.savefig(outputPath + str(imageidx).zfill(ndigits) + '-histograms.png')
    imageidx += 1

    # density
    print("density")
    dataframe.plot(kind='density', subplots=True, layout=(6,7), sharex=False, legend=False)
    plt.savefig(outputPath + str(imageidx).zfill(ndigits) + '-density.png')
    imageidx += 1

    # box and whisker plots
    print("box and whisker plots")
    dataframe.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False)
    plt.savefig(outputPath + str(imageidx).zfill(ndigits) + '-box.png')
    imageidx += 1

    # scatter plot matrix
    print("scatter plot matrix")
    scatter_matrix(dataframe)
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(ndigits) + '-scatter-plot.png')
    imageidx += 1

    # correlation matrix
    print("correlation matrix")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    #plt.show()
    plt.savefig(outputPath + str(imageidx).zfill(ndigits) + '-correlation-matrix.png')
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
    plt.savefig(outputPath + str(imageidx).zfill(ndigits) + '-standardized-histograms.png')
    imageidx += 1

    
def run(inputPath='../input/', outputPath='../output/'):
    print '<<< Running Exploratory Data Analysis #1 ==='

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)    
        
    # Load dataset
    dataframe = loadDataframe(inputPath + 'curso_1200.csv')
    dataframe = dataCleansing(dataframe)
    
    descriptiveStatistics(dataframe, outputPath)
    dataVisualizations(dataframe, outputPath)
    
    
    
    print '=== Running Exploratory Data Analysis #1 >>>'
    
    
    

