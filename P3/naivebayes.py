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

def part1():


    pos = {}
    neg = {}
    word = ["excellent", "terrible", "awful"]
    for w in word:
        pos[w] = 0
        neg[w] = 0
    
    for file in os.listdir('txt_sentoken/pos'):
            f = open('txt_sentoken/pos/' + file)
            f = re.split('\W+', f.read().lower())
            f.remove('')
            f = list(OrderedDict.fromkeys(f))
            for w in word:
                if w in f:
                    pos[w] += 1
    
    for file in os.listdir('txt_sentoken/neg'):
            f = open('txt_sentoken/neg/' + file)
            f = re.split('\W+', f.read().lower())
            f.remove('')
            f = list(OrderedDict.fromkeys(f))
            for w in word:
                if w in f:
                    neg[w] += 1
    
    print(pos)
    print(neg)


def randomly_sepfolders():
    sets = ['train','valid','test']
    sets2 = ['pos','neg']
    
    for se in sets:
        for se2 in sets2:
            txt_file_name = se + '/' + se2
            if not os.path.exists(txt_file_name):
                os.makedirs(txt_file_name) 
                
    pos = "txt_sentoken/pos"
    neg = "txt_sentoken/neg"

    np.random.seed(0)
    num = np.random.permutation(1000)   
    
    train = num[0:600]
    valid = num[600:800]
    test = num[800:1000]

    for a in range(0,600):
        copy2('txt_sentoken/pos/' + os.listdir(pos)[train[a]],'train/pos')
        copy2('txt_sentoken/neg/'+ os.listdir(neg)[train[a]],'train/neg')
    for b in range(0,200):
        copy2('txt_sentoken/pos/' + os.listdir(pos)[valid[b]],'valid/pos')
        copy2('txt_sentoken/neg/'+ os.listdir(neg)[valid[b]],'valid/neg')    
    for c in range(0,200):
        copy2('txt_sentoken/pos/' + os.listdir(pos)[test[c]],'test/pos')
        copy2('txt_sentoken/neg/' + os.listdir(neg)[test[c]],'test/neg')
 

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




def get_wordcount(path):
    pos = {}
    neg = {}
    
    poscount = 0
    negcount = 0
    pos_train, pos_valid, pos_test, neg_train, neg_valid, neg_test = counsdata(trainnum = 600,validnum = 200,testnum = 200)

    if fnmatch.fnmatch(path, '*'+'train'+'*'):
        tempos = pos_train
        temneg = neg_train
    if fnmatch.fnmatch(path, '*'+'valid'+'*'):
        tempos = pos_valid
        temneg = neg_valid
    if fnmatch.fnmatch(path, '*'+'test'+'*'):
        tempos = pos_test
        temneg = neg_test
    
    for f in tempos:
        length= len(f)
        for word in f:
            if word not in pos:
                pos[word] = 1.0/float(length)
            else:
                pos[word] += 1.0/float(length)
            poscount += 1.0/float(length)
    for f in temneg:
        length= len(f)
        for word in f:
            if word not in neg:
                neg[word] = 1.0/float(length)
            else:
                neg[word] += 1.0/float(length)
            negcount += 1.0/float(length)
    return pos, neg, poscount, negcount


def prob_C_given_word(cls, word, m, k):
    word_list = [word]
    prob_word = math.log(totaltrain[word]/float(traintotal_count))
    prob_word_given_C = log_prob_C_given_words(word_list, m, k, cls)
    return prob_word_given_C - prob_word


def log_prob_C_given_words(word_list, m, k, cls):
    prob_words_cls = add_logs_prob_words_given_C(word_list, m, k, cls)
    if cls == 1:
        prob_C = p_pos
    else:
        prob_C = p_neg
    return prob_words_cls + math.log(prob_C)
    
    
def add_logs_prob_words_given_C(word_list, m, k, cls):
    sumlog = 0
    for word in word_list:
        if cls == 1:
            if word in postrain:
                word_count = postrain[word]
            else:
                word_count = 0
            prob_word = math.log((word_count + m * k)/float((postrain_count + k)))
        else:
            if word in negtrain:
                word_count = negtrain[word]
            else:
                word_count = 0
            prob_word = math.log((word_count + m * k)/float((negtrain_count + k)))
        sumlog += prob_word
    return sumlog


def predict_review(word_list, m, k):
    prob_pos_review = log_prob_C_given_words(word_list, m, k, 1)
    prob_neg_review = log_prob_C_given_words(word_list, m, k, 0)
    if prob_pos_review >= prob_neg_review:
        return 1
    else:
        return 0

def get_performance(path, m, k):
    pos_countp = 0
    neg_countp = 0
    if fnmatch.fnmatch(path, '*'+'train'+'*'):
        tempos = pos_train
        temneg = neg_train
    if fnmatch.fnmatch(path, '*'+'valid'+'*'):
        tempos = pos_valid
        temneg = neg_valid
    if fnmatch.fnmatch(path, '*'+'test'+'*'):
        tempos = pos_test
        temneg = neg_test
    totalpos = len(tempos)
    totalneg = len(temneg)
    wrongcount = 0
    for f in tempos:
        predict = predict_review(f, m, k)
        if (predict == 1):
            pos_countp += 1
        else: wrongcount += 1
    print (path+" positive performance: "+str(pos_countp*100/float(totalpos))+"%")
    for f in temneg:
        predict = predict_review(f, m, k)
        if (predict == 0):
            neg_countp += 1
        else: wrongcount += 1
    print(neg_countp, pos_countp, totalpos,totalneg,wrongcount)
    print (path+" negative performance: "+str(neg_countp*100/float(totalneg))+"%")
    print (path+" total performance: "+ str((pos_countp+neg_countp) * 100 / float(totalpos + totalneg))+"%")




def part2():
    global postrain, negtrain, totaltrain, traintotal_count, postrain_count, negtrain_count, p_pos, p_neg
    
   
    postrain, negtrain, postrain_count, negtrain_count = get_wordcount('train')
    
    traintotal_count = postrain_count + negtrain_count
    totaltrain = {k: postrain.get(k, 0) + negtrain.get(k, 0) for k in set(postrain) | set(negtrain)}
        
    p_pos = postrain_count / float(traintotal_count)
    p_neg = negtrain_count / float(traintotal_count)

    m = 3
    k = 20

    train_per = get_performance('train', m, k)
    val_per = get_performance('valid', m, k)
    test_per = get_performance('test', m, k)
    
    h_pos = []
    h_neg = []
    
    for word in postrain:
        prob_pos_given_word = prob_C_given_word(1, word, m, k)
        heapq.heappush(h_pos, (prob_pos_given_word, word))
    
    for word in negtrain:
        prob_neg_given_word = prob_C_given_word(0, word, m, k)
        heapq.heappush(h_neg, (prob_neg_given_word, word))
        
    top10_pos = heapq.nlargest(10, h_pos)
    top10_neg = heapq.nlargest(10, h_neg)
    print 'top 10 pos:', ([voca[1] for voca in top10_pos])
    print 'top 10 neg:', ([voca[1] for voca in top10_neg])



part2()
