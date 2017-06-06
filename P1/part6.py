import numpy as np
from numpy import *
import pandas as pd
from numpy.linalg import norm
import os
from os.path import isfile, join
from rgb2gray import rgb2gray
import fnmatch
from scipy.misc import imread
from scipy.misc import imsave
from numpy.linalg import inv
from part3 import f, df, grad_descent
from numpy import linalg as LA

def constructmatrix():
    x = np.zeros(shape=(600,1024))
    y = np.zeros(shape=(600,6))
    l = []
    i = 0
    for f in os.listdir('part6/training'):
        name = 'part6/training/' + f
        try:
            im = imread(name)
            if fnmatch.fnmatch(f, '*'+'drescher'+'*'):
                img = rgb2gray(im)
                a = np.ravel(img)
                x[i] = a
                y[i] = [1,0,0,0,0,0]
            if fnmatch.fnmatch(f, '*'+'ferrera'+'*'):
                img = rgb2gray(im)
                a = np.ravel(img)
                x[i] = a
                y[i] = [0,1,0,0,0,0]
            if fnmatch.fnmatch(f, '*'+'chenoweth'+'*'):
                img = rgb2gray(im)
                a = np.ravel(img)
                x[i] = a
                y[i] = [0,0,1,0,0,0]
            if fnmatch.fnmatch(f, '*'+'baldwi'+'*'):
                img = rgb2gray(im)
                a = np.ravel(img)
                x[i] = a
                y[i] = [0,0,0,1,0,0]
            if fnmatch.fnmatch(f, '*'+'hader'+'*'):
                img = rgb2gray(im)
                a = np.ravel(img)
                x[i] = a
                y[i] = [0,0,0,0,1,0]
            if fnmatch.fnmatch(f, '*'+'carell'+'*'):
                img = rgb2gray(im)
                a = np.ravel(img)
                x[i] = a
                y[i] = [0,0,0,0,0,1]
            i += 1
        except:
             print("not an image file")
       
    l.append(x)
    l.append(y)
    return l


def f(x, y, theta):
    x = np.transpose(x)
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(x.T,theta.T)) ** 2)

def df(x, y, theta):
    x = np.transpose(x)
    x = vstack( (ones((1, x.shape[1])), x))
    return 2*(dot(x,(dot(theta,x)).T-y)).T

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-10   
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    ite  = 0
    while norm(t - prev_t) >  EPS and ite < max_iter:
        prev_t = t.copy()
        t = t - alpha*df(x, y, t)
        if ite % 5000 == 0:
            print "Iter", ite
            print "t" ,t
            print  (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        ite += 1
    return t



def trainpart6():
    l = constructmatrix()
    theta0 = np.zeros(shape=(6,1025))
    theta = grad_descent(f, df, l[0], l[1], theta0, 0.0000010)
    return theta

def valadationpart6(theta,valortest):
    lst = []
    m = 0
    n = 0
    for f in os.listdir('part6/' + valortest):
        name = 'part6/' + valortest + '/' + f
        try:
            im = imread(name)
            img = rgb2gray(im)
            a = np.ravel(img)
            a = np.insert(a, 0, 1)
            x = dot(theta,a)
            posi = np.argmax(x)
            if posi == 0 and ('drescher' in name):
                m = m+1
            elif posi == 1 and ('ferrera' in name):
                m = m+1
            elif posi == 2 and ('chenoweth' in name):
                m = m+1
            elif posi == 3 and ('baldwin' in name):
                m = m+1
            elif posi == 4 and ('hader' in name):
                m = m+1
            elif posi == 5 and ('carell' in name):
                m = m+1
            else:
                n = n + 1
        except:
            print("not an image file")
    
    print (float(m)/(float(m)+float(n)))

def savetheta(theta):
    act = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell' ]
    i = 0           
    while i < 6:
       lst = theta[i]
       lst = np.delete(theta[i],0)
       lst = np.reshape(lst,(32,32))
       imsave("part6imag " +act[i]+'.jpg', lst)
       i = i+1
    
    
