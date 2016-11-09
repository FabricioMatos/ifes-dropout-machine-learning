# License: BSD 3 clause <https://opensource.org/licenses/BSD-3-Clause>
# Copyright (c) 2016, Fabricio Vargas Matos <fabriciovargasmatos@gmail.com>
# All rights reserved.

'''
Project: IFES dropout prediction

Use Machine Learning techniques to model a student's dropout prediction system for Brazilian Federal Institutes of Education, Science and Technology analyzing data available in the Q-Academico information system.

After exploring many algorithms and data preparing techniques
'''

# Load libraries
import pandas
import numpy
import lib.ifes_graduate_diploma_ml_model as ifes_model

def train(inputname):
    #set the path to the in/out files
    inputFilePath = '../input/' + inputname + '.csv'
    outputFileNameForModel = 'output/' + inputname + '_LR_model.sav'

    #train the classification model and save it to outputFileNameForModel
    ifes_model.trainStudentDropoutPrediction(inputFilePath, outputFileNameForModel)


def predict(inputname, modelname):    
    print 
    print '###############################################'
    print '#########  Student Dropout Prediction #########'
    print '###############################################'
    print 'Params: inputname="%s" modelname="%s"' % (inputname, modelname)
    print
    
    modelFileName = 'output/' + inputname+ '_LR_model.sav'
    inputFileName = '../input/' + inputname + '.csv'
    outputFileName = 'output/' + inputname + '_predictions.csv'

    # predict and return a matrix with ['hash_cod_matricula', 'dropout-predicion']
    dataframe = ifes_model.predictFromFile(modelFileName, inputFileName, outputFileName)
    
    print dataframe.groupby('dropout-predicion').size()
    

        
train(inputname='curso_1200')
train(inputname='curso_2770')

predict(inputname='curso_1200', modelname='curso_1200')
predict(inputname='curso_2770', modelname='curso_2770')
predict(inputname='curso_2770', modelname='curso_1200')
predict(inputname='curso_1200', modelname='curso_2770')

