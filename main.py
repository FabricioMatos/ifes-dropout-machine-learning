# License: BSD 3 clause <https://opensource.org/licenses/BSD-3-Clause>
# Copyright (c) 2016, Fabricio Vargas Matos <fabriciovargasmatos@gmail.com>
# All rights reserved.

'''
Project: IFES dropout prediction

Use Machine Learning techniques to model a student's dropout prediction system for Brazilian Federal Institutes of Education, Science and Technology analyzing data available in the Q-Academico information system.

After exploring many algorithms and data preparing techniques
'''

# Load libraries
import lib.ifes_graduate_diploma_ml_model as ifes_model

def train():
    inputname = 'curso_1200' 
    #inputname = 'curso_2770' 

    #set the path to the in/out files
    inputFilePath = '../input/' + inputname + '.csv'
    outputFileNameForModel = 'output/' + inputname + '_LR_model.sav'

    #train the classification model and save it to outputFileNameForModel
    ifes_model.trainStudentDropoutPrediction(inputFilePath, outputFileNameForModel)


def predict():    
    
    modelFileName = 'output/curso_1200_LR_model.sav'
    inputFileName = '../input/curso_2770.csv'

    # predict
    prediction = ifes_model.predictFromFile(modelFileName, inputFileName)

    print prediction
        
        
#train()
predict()

