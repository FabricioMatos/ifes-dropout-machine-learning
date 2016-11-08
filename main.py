# License: BSD 3 clause <https://opensource.org/licenses/BSD-3-Clause>
# Copyright (c) 2016, Fabricio Vargas Matos <fabriciovargasmatos@gmail.com>
# All rights reserved.

'''
Project: IFES dropout prediction

Use Machine Learning techniques to model a student's dropout prediction system for Brazilian Federal Institutes of Education, Science and Technology analyzing data available in the Q-Academico information system.
'''


# Load libraries
import lib.eda1 as eda1
import lib.eda2 as eda2
import lib.eda3 as eda3

#debug flags
#createImagesFlag = False
createImagesFlag = True


def run(inputname):
    inputFilePath = '../input/' + inputname + '.csv'    
    
    #run a first exploratory data analyses
    eda1.run(inputFilePath, 'output/' + inputname + '/eda1/', createImagesFlag)

    #drop "not fair" features identified in the eda1
    eda2.run(inputFilePath, 'output/' + inputname + '/eda2/', createImagesFlag)

    #try to improve the preliminar results appling "feature selection" techniques
    eda3.run(inputFilePath, 'output/' + inputname + '/eda3/', createImagesFlag)
    


run('curso_1200')   #Bacharelado em Sistemas de Informacao - Campus Serra
#run('curso_2770')   #Tecnologia em Analise e Desenvolvimento de Sistemas EAD - Campus Serra

#run('curso_360')    #Tecnico em Informatica - Campus Serra



