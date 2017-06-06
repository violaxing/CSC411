import numpy as np
from numpy import *
import pandas as pd
from numpy.linalg import norm
import os
from os.path import isfile, join
from rgb2gray import rgb2gray
import fnmatch
from scipy.misc import imread
from numpy.linalg import inv
from part3 import *
from numpy import linalg as LA
from part5 import *
from part6 import *
from download import *
from randomselect import *
from rgb2gray import *


#part1
#Note: you need to create the uncropped folder first in order 
#for this to work
def download():
    downloadimage()

#part2
#Note: you need to create the part3 folder and training, test and
#validaiton folder under the part3 folder first in order 
#for this to work
def part2seper():
    deleteallimagepart2()
    randompart2()


def part3():
    theta = trainpart3()
    testpart3(theta,"validation")
    testpart3(theta,"test")
    
#part4
#Note: you need to create the part4 folder first in order 
#for this to work
def part4():
    deleteallimagepart4()
    randompart4()
    theta = trainpart4()
    theta = np.delete(theta,0)
    theta = np.reshape(theta,(32,32))
    imsave("part4pic1.jpg", theta)
    theta2 = trainpart3()
    theta2 = np.delete(theta2,0)
    theta2 = np.reshape(theta2,(32,32))
    imsave("part4pic2.jpg", theta2 )

#part4
#Note: you need to create the part5 folder. And then and training and validation folder
# under part5 folder. And female and male folder under the training folder.
def part5():
    deleteallimagepart5()
    randompart5()
    trainpart5()

#part4
#Note: you need to create the part6 folder. And then and training, validation
#   and test folder under part6 folder. 

def part6():
    deleteallimagepart6()
    randompart6()
    theta = trainpart6()
    valadationpart6(theta,"validation")
    valadationpart6(theta,"test")
    savetheta(theta)
    
