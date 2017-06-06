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
from numpy import linalg as LA


def f(x, y, theta):
    x = np.transpose(x)
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = np.transpose(x)
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10   
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    ite  = 0
    while norm(t - prev_t) >  EPS and ite < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if ite % 5000 == 0:
            print "Iter", ite
            print "Gradient: ", df(x, y, t), "\n"
        ite += 1
    return t

def constructpart3():
    lst = []
    x = np.zeros(shape=(200,1024))
    y = np.zeros(200)
    i = 0
    for f in os.listdir('part3/training'):
        name = 'part3/training/' + f
        if fnmatch.fnmatch(f, '*'+'hader'+'*'):
           im = imread(name)
           img = rgb2gray(im)
           a = np.ravel(img)
           x[i] = a
           y[i] = 1
           i += 1
        if fnmatch.fnmatch(f, '*'+'carell'+'*'):      
           im = imread(name)
           img = rgb2gray(im)
           a = np.ravel(img)
           x[i] = a
           y[i] = 0
           i += 1
    lst.append(x)
    lst.append(y)
    return lst

def trainpart3():
    lst =  constructpart3()
    theta0 = np.zeros(1025)
    x = lst[0]
    y = lst[1]
    theta = grad_descent(f, df, x, y, theta0, 0.0000010)
    return theta



#validortest is a string "validation" or "test"
def testpart3(theta,validortest):
    i = 0
    j = 0
    for f in os.listdir('part3/'+validortest):
        name = 'part3/' + validortest + '/' + f
        if fnmatch.fnmatch(f, '*'+'a'+'*'):
            im = imread(name)
        im = rgb2gray(im)
        a = np.ravel(im)
        a = np.insert(a, 0, 1)
        x = dot(theta.T,a)
        if LA.norm(x)> 0.5 and ('hader' in name):
                i = i+1
        elif LA.norm(x)> 0.5 and ('hader' not in name):
                j = j+1
        elif LA.norm(x)< 0.5 and ('carell' not in name):
                j = j+1
        elif LA.norm(x)< 0.5 and ('carell' in name):
                i = i+1
    k = float(i)/(float(i) + float(j))
    print k
    return k



#validortest is a string "validation" or "test"

def costvalue(theta,validortest):
    lst = []
    x = np.zeros(shape=(20,1024))
    y = np.zeros(20)
    i = 0
    for f in os.listdir('part3/'+validortest):
        name = 'part3/' + validortest + '/' + f
        if fnmatch.fnmatch(f, '*'+'hader'+'*'):
           im = imread(name)
           img = rgb2gray(im)
           a = np.ravel(img)
           x[i] = a
           y[i] = 1
           i += 1
        if fnmatch.fnmatch(f, '*'+'carell'+'*'):      
           im = imread(name)
           img = rgb2gray(im)
           a = np.ravel(img)
           x[i] = a
           y[i] = 0
           i += 1
    x = np.transpose(x)
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

                        
def constructpart4():
    lst = []
    x = np.zeros(shape=(4,1024))
    y = np.zeros(4)
    i = 0
    j = 0
    k = 0
    for f in os.listdir('part4'):
        name = 'part4/' + f
        if fnmatch.fnmatch(f, '*'+'hader'+'*'):
            im = imread(name)
            img = rgb2gray(im)
            a = np.ravel(img)
            x[i] = a
            y[i] = 1
            i += 1
        if fnmatch.fnmatch(f, '*'+'carell'+'*'):
            im = imread(name)
            img = rgb2gray(im)
            a = np.ravel(img)
            x[i] = a
            y[i] = 0
            i += 1

    lst.append(x)
    lst.append(y)
    return lst


def trainpart4():
    lst =  constructpart4()
    theta0 = np.zeros(1025)
    print lst[0]
    print lst[1]
    theta = grad_descent(f, df, lst[0], lst[1], theta0, 0.0000010)
    print theta.size
    return theta
