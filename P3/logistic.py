import numpy as np
from numpy import *
import pandas as pd
from numpy.linalg import norm
import os
from os.path import isfile, join
import fnmatch
from scipy.misc import imread
from numpy.linalg import inv
from collections import Counter
from collections import OrderedDict
import math
from shutil import copy2
import re
import heapq
import operator


def counsdata(trainnum = 600,validnum = 200,testnum = 200):
    global pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test
    pos_train = []
    pos_valid= []
    pos_test= []
    neg_train= []
    neg_valid= []
    neg_test= []

    i = 0
    for filename in os.listdir("train/pos/"):
        while i < trainnum:
            f = open("train/pos/"+filename, "r")
            text = re.split('\W+', f.read().lower())
            pos_train.append(text)
            i += 1
    i = 0
    for filename in os.listdir("valid/pos/"):
        while i < validnum:
            f = open("valid/pos/"+filename, "r")
            text = re.split('\W+', f.read().lower())
            pos_valid.append(text)
            i += 1
    i = 0
    for filename in os.listdir("test/pos/"):
        while i < testnum:
            f = open("test/pos/"+filename, "r")
            text = re.split('\W+', f.read().lower())
            pos_test.append(text)
            i += 1
    i = 0
    for filename in os.listdir("train/neg/"):
        while i < trainnum:
            f = open("train/neg/"+filename, "r")
            text = re.split('\W+', f.read().lower())
            neg_train.append(text)
            i += 1
    i = 0
    for filename in os.listdir("valid/neg/"):
        while i < validnum:
            f = open("valid/neg/"+filename, "r")
            text = re.split('\W+', f.read().lower())
            neg_valid.append(text)
            i += 1
    i = 0
    for filename in os.listdir("test/neg/"):
        while i < testnum:
            f = open("test/neg/"+filename, "r")
            text = re.split('\W+', f.read().lower())
            neg_test.append(text)
            i += 1           

    return pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test







def constructX(num,pos,neg,dic):
    X = None
    count = 0
    tem = pos + neg
    for f in tem:
        word = np.zeros(num)
        for w in f:
            word[dic[w]] = 1
        if X is None:
            X = word.copy()
        else:
            X = np.vstack((X,word))
        count +=1
    x = np.insert(X,0,1,axis = 1)
    print(count)
    return X

def constructY(train = 600, validation = 200,test= 200):
    Ytrain = np.zeros((train*2,2))
    Ytrain[0:train,0].fill(1)
    Ytrain[train:(train*2),1].fill(1)
    Yvalid = np.zeros((validation*2,2))
    Yvalid[0:validation,0].fill(1)
    Yvalid[validation:validation*2,1].fill(1)
    Ytest = np.zeros((test*2,2))
    Ytest[0:test,0].fill(1)
    Ytest[test:test*2,1].fill(1)
    return Ytrain, Yvalid, Ytest


def part4():
    pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test = counsdata(trainnum = 600,validnum = 200,testnum = 200)
    dataset = pos_train + pos_valid + pos_test+ neg_train +neg_valid+ neg_test
    dic = {}
    num = 0
    for f in dataset:
        for word in f:
            if word not in dic:
                dic[word] = num
                num = num + 1

    Xtrain = constructX(39444,pos_train,neg_train,dic)
    Xvalid = constructX(39444,pos_valid,neg_valid,dic)
    Xtest = constructX(39444,pos_test,neg_test,dic)   
    Ytrain, Yvalid, Ytest = constructY(train = 600, validation = 200,test= 200)
 
    W = np.random.normal(0., 1e-5, size=(Xtrain.shape[1],Ytrain.shape[1]))
    train_regression(W,Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest, decay_rate = 0.999)


def train_regression(W , X , Y, X_valid, Y_valid, X_test, Y_test, epoch = 800, decay_rate = 0.9, alpha = 1e-2):
    x_axis = np.linspace(0,epoch,epoch/float(100))
    y = np.zeros(x_axis.shape[0])
    z = np.zeros(x_axis.shape[0])
    w = np.zeros(x_axis.shape[0])
    for epochs in range(epoch):
        (O,P) = forward(X,W)
        W = W * decay_rate - alpha*df(X,O,P,Y)
        if epochs % 10 == 0:
            train_acc = calc_accuracy(P,Y)
            train_cost = cost_function(X,W,Y)
            (O_valid,P_valid) = forward(X_valid,W)
            valid_acc = calc_accuracy(P_valid,Y_valid)
            valid_cost = cost_function(X_valid,W,Y_valid)
            (O_test,P_test) = forward(X_test,W)
            test_acc = calc_accuracy(P_test,Y_test)
            test_cost = cost_function(X_test,W,Y_test)

            y[epochs/100] = train_acc
            z[epochs/100] = test_acc
            w[epochs/100] = valid_acc
            print 'Epoch {}; train_acc={:1.5f}, train_cost={:1.5f}, test_acc={:1.5f}, test_cost={:1.5f}, valid_acc = {:1.5f}'.format(epochs,train_acc,train_cost,test_acc,test_cost,valid_acc)   
    training_set_accuracy, = plt.plot(x_axis, y, 'r-',label = 'training set error rate')
    test_set_accuracy, = plt.plot(x_axis, z, 'g-',label = 'test set error rate')
    valid_set_accuracy, = plt.plot(x_axis, w, 'b-',label = 'validation set error rate')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(handler_map={training_set_accuracy: HandlerLine2D(numpoints=4)})
    plt.title('Multinomial Logistic Regression')
    plt.show()

def cost_function(X,W,Y):
    (O,P) = forward(X,W)
    return -sum((Y*log(P)))/X.shape[0]    

def forward(X, W):
    O = np.dot(X,W)
    P = softmax(O)
    return (O,P)

def softmax(y):
    y = y.T
    return (exp(y)/tile(sum(exp(y),0), (len(y),1))).T

    
def df(X,O,P,y):
    return np.dot((X.T),(P-y))/X.shape[0]
    
def calc_accuracy(P,y):
    P = (P == P.max(axis=1)[:,None]).astype(int)
    count = 0
    for j in range(P.shape[0]):
        flag = 1
        for i in range(P.shape[1]):
            if P[j,i] != y[j,i]:
                flag = 0
        count += flag
    return float(count)/y.shape[0]

part4()
