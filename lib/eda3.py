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

import lib.eda1 as eda1

#global variables
start = time.clock()

def duration():
    global start
    
    end = time.clock()
    print '\nDuration: %.2f ' % (end - start)
    
    start = time.clock()


#drop not interesting columns and fill NaN values
def dataCleansing(dataframe):
    #axis: 0 for rows and 1 for columns
    dataframe.drop('cep', axis=1, inplace=True)
    dataframe.drop('sit_matricula', axis=1, inplace=True)

    #drop attributes impacted for the fact that the student isn't sduding anymore.
    #we want attributes that would be the same for students starting the second semester
    dataframe.drop('periodo_atual', axis=1, inplace=True)
    dataframe.drop('coeficiente_rendimento', axis=1, inplace=True)
    dataframe.drop('coeficiente_progressao', axis=1, inplace=True)
    dataframe.drop('reprovacoes_por_nota', axis=1, inplace=True)
    dataframe.drop('reprovacoes_por_falta', axis=1, inplace=True)
    dataframe.drop('aprovacoes', axis=1, inplace=True)
    dataframe.drop('aproveitamentos', axis=1, inplace=True)

    
    #drop attributes with low relevance
    dataframe.drop('estado_civil', axis=1, inplace=True)
    dataframe.drop('sexo', axis=1, inplace=True)
    dataframe.drop('possui_filhos', axis=1, inplace=True)
    
    
    #replace NaN with 0
    dataframe.fillna(value=0, inplace=True)
    
    
    return dataframe
    
    
    
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
    
    dataframe = dataCleansing(dataframe)
    
    # Understand the data
    #eda1.descriptiveStatistics(dataframe, outputPath)
    eda1.dataVisualizations(dataframe, outputPath)
        
    #Split-out train/validation dataset
    X_train, X_validation, Y_train, Y_validation = eda1.splitoutValidationDataset(dataframe)
    
    # Select the most effective features
    eda1.featureSelection(dataframe.dtypes.keys(), X_train, Y_train)    
    
    # Evaluate Algorithms
    eda1.evaluteAlgorithms(X_train, Y_train, outputPath)
    
    # Standardize the dataset and reevaluate the same algorithms
    eda1.standardizeDataAndReevaluateAlgorithms(X_train, Y_train, outputPath)
    
    # Evaluate Ensemble Algorithms
    eda1.evaluateEnsembleAlgorith(X_train, Y_train, outputPath)
    
    duration()    
    print '<<< THEN END - Running Exploratory Data Analysis #3 >>>'
    
