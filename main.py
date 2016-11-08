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

print '<<< main ==='

#debug flags
createImagesFlag = True
#createImagesFlag = False



inputFilePath = '../input/curso_1200.csv'

#run exploratory data analyses
eda1.run(inputFilePath, 'output/eda1/', createImagesFlag)
eda2.run(inputFilePath, 'output/eda2/', createImagesFlag)
#eda3.run(inputFilePath, 'output/eda3/', createImagesFlag)

print '=== main >>>'