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
import lib.eda4 as eda4

#debug flags
createImagesFlag = False
createImagesFlag = True


def run(inputname, dropColumns):
    inputFilePath = '../input/' + inputname + '.csv'    
    
    #run a first exploratory data analyses
    eda1.run(inputFilePath, 'output/' + inputname + '/eda1/', createImagesFlag, dropColumns)

    #try to improve the preliminar results appling "feature selection" techniques
    eda2.run(inputFilePath, 'output/' + inputname + '/eda2/', createImagesFlag, dropColumns)
    
    #tune the 3 best aglorithms (courses 1200 and 2770)
    #eda3.run(inputFilePath, 'output/' + inputname + '/eda3/', createImagesFlag, dropColumns)

    #tune the 3 best aglorithms (everyone from 2009 to 2014)
    eda4.run(inputFilePath, 'output/' + inputname + '/eda4/', createImagesFlag, dropColumns)
    

#Bacharelado em Sistemas de Informacao - Campus Serra
#run('curso_1200', dropColumns=['cod_instituicao', 'cod_curso'])   

#Tecnologia em Analise e Desenvolvimento de Sistemas EAD - Campus Serra
#run('curso_2770', dropColumns=['cod_instituicao', 'cod_curso'])   

#everybody from 2009 to 2014
run('ifes_2009_to_2014', dropColumns=[]) 


