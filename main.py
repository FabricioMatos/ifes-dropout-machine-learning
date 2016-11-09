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

def train(inputname, modelname):
    #set the path to the in/out files
    inputFilePath = '../input/%s.csv' % inputname
    outputFileNameForModel = 'output/%s_LR.joblib' % modelname

    #train the classification model and save it to outputFileNameForModel
    ifes_model.trainStudentDropoutPrediction(inputFilePath, outputFileNameForModel)


def predict(inputname, modelname):    
    print 
    print '###############################################'
    print '#########  Student Dropout Prediction #########'
    print '###############################################'
    print 'Params: inputname="%s" modelname="%s"' % (inputname, modelname)
    print
    
    modelFileName = 'output/%s_LR.joblib' % modelname
    inputFileName = '../input/%s.csv' % inputname
    outputFileName = 'output/%s_%s_predictions.csv' % (inputname, modelname)

    # predict and return a matrix with ['hash_cod_matricula', 'dropout-predicion']
    dataframe = ifes_model.predictFromFile(modelFileName, inputFileName, outputFileName)
    
    print dataframe.groupby('dropout-predicion').size()
    
    ifes_model.checkPredictionAccuracy(dataframe, inputFileName)
    

        
train(inputname='curso_1200', modelname='model_1200')
train(inputname='curso_2770', modelname='model_2770')

predict(inputname='curso_1200', modelname='model_1200')
predict(inputname='curso_2770', modelname='model_2770')
predict(inputname='curso_2770', modelname='model_1200')
predict(inputname='curso_1200', modelname='model_2770')

