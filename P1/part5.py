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
from part3 import f, df, grad_descent
from numpy import linalg as LA
import matplotlib.pyplot as plt

def constructxy(k):
    l = []
    x = np.zeros(shape=(2*k,1024))
    y = np.zeros(2*k)
    i = 0
    for f in os.listdir('part5/training/female'):
        if i >= k:
            break
        name = 'part5/training/female/' + f
        if fnmatch.fnmatch(f, '*'+'jp'+'*'):
            im = imread(name)
            img = rgb2gray(im)
            a = np.ravel(img)
            x[i] = a
            y[i] = 1
            i += 1       
    i = 0           
    for f in os.listdir('part5/training/male'):
        if i >= k:
            break
        name = 'part5/training/male/' + f
        if fnmatch.fnmatch(f, '*'+'jp'+'*'):
            im = imread(name)
            img = rgb2gray(im)
            a = np.ravel(img)
            x[i+k] = a
            y[i+k] = 0
            i += 1
    l.append(x)
    l.append(y)
    return l

def trainpart5():   

    j = [5,10,50,100,150,200,250,300]
    lstone = []
    lsttwo = []
    lst = []
    for k in j:
        x = constructxy(k)[0]
        y = constructxy(k)[1]
        theta0 = np.zeros(1025)
        theta = grad_descent(f, df, x, y, theta0, 0.0000010)
        m = 0
        n = 0
        for fi in os.listdir('part5/validation'):
              name = 'part5/validation/' + fi
              try:
                  im = imread(name)
                  img = rgb2gray(im)
                  a = np.ravel(img)
                  a = np.insert(a, 0, 1)
                  x = dot(theta.T,a)
                  if x> 0.5 and (('bracco'in name) or ('gilpin' in name) or ('harmon'in name)):
                      m = m+1
                  elif x> 0.5 and (not (('bracco'in name) or ('gilpin' in name) or ('harmon'in name))):
                      n = n+1
                  elif x< 0.5 and (not(('bulter'in name) or ('radcliffe' in name) or ('vartan'in name))):
                      n = n+1
                  elif x< 0.5 and (('bulter'in name) or ('radcliffe' in name) or ('vartan'in name)):
                      m = m+1
              except:
                print("not an image file")
        lstone.append(m)
        lsttwo.append(n)
        print lstone
        print lsttwo
        lst.append(float(m)/(float(m)+float(n)))
    k = [10,20,100,200,300,400,500,600]
    plt.plot(k,lst)
    plt.ylabel('performance')
    plt.show()
    plt.save("performance.jpg")
