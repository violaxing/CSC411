import os
import random
import numpy as np
from scipy.misc import *
from rgb2gray import rgb2gray

def randompart2():
    act = ["hader","carell"]
    for name in act:
        lst = []
        for x in os.listdir("cropped"):
            if name in x:
                lst.append(x)
        a = np.random.choice(lst, 120, replace=False)
        train = a[:100]
        test = a[100:110]
        vali = a[110:120]
        for tr in train :
            im = imread("cropped/"+ tr)
            img = rgb2gray(im)
            imsave("part3/training/"+tr, im)
        for te in test:
            im = imread("cropped/"+ te)
            img = rgb2gray(im)
            imsave("part3/test/"+te, im)
        for va in vali :
            im = imread("cropped/"+ va)
            img = rgb2gray(im)
            imsave("part3/validation/"+va, im)
        
        
    
def deleteallimagepart2():
    for f in os.listdir("part3/training"):
        os.remove("part3/training/"+f)
    for f in os.listdir("part3/test"):
        os.remove("part3/test/"+f)
    for f in os.listdir("part3/validation"):
        os.remove("part3/validation/"+f)



def randompart4():
    act = ["hader","carell"]
    for name in act:
        lst = []
        for x in os.listdir("cropped"):
            if name in x:
                lst.append(x)
        a = np.random.choice(lst, 2, replace=False)
        for pic in a:
            im = imread("cropped/"+ pic)
            img = rgb2gray(im)
            imsave("part4/"+pic, im)

            
def deleteallimagepart4():
    for f in os.listdir("part4"):
        os.remove("part4/"+f)


def randompart5():
    act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
    acto = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
    act_test = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan',
     'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
    for na in act:
        name = na.split()[1].lower()
        lst = []
        for x in os.listdir("cropped"):
            if name in x:
                lst.append(x)
        a = np.random.choice(lst, 100, replace=False)

        for tr in a :
            im = imread("cropped/"+ tr)
            img = rgb2gray(im)
            imsave("part5/training/female/"+tr, im)
    for na in acto:
        name = na.split()[1].lower()
        lst = []
        for x in os.listdir("cropped"):
            if name in x:
                lst.append(x)
        a = np.random.choice(lst, 100, replace=False)

        for tr in a :
            im = imread("cropped/"+ tr)
            img = rgb2gray(im)
            imsave("part5/training/male/"+tr, im)
    
    for tes in act_test:
        test = tes.split()[1].lower()
        lsttest = []
        for x in os.listdir("cropped"):
            if test in x:
                lsttest.append(x)
        b = np.random.choice(lsttest, 10, replace=False)

        for tr in b:
            im = imread("cropped/"+ tr)
            img = rgb2gray(im)
            imsave("part5/validation/"+tr, im)

def deleteallimagepart5():
    for f in os.listdir("part5/training/male"):
        os.remove("part5/training/male/"+f)
    for f in os.listdir("part5/training/female"):
        os.remove("part5/training/female/"+f)
    for f in os.listdir("part5/validation/"):
        os.remove("part5/validation/"+f)

def randompart6():
    act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth',
          'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    for na in act:
        name = na.split()[1].lower()
        lst = []
        for x in os.listdir("cropped"):
            if name in x:
                lst.append(x)
        a = np.random.choice(lst, 120, replace=False)
        train = a[:100]
        test = a[100:110]
        vali = a[110:120]
        for tr in train :
            im = imread("cropped/"+ tr)
            img = rgb2gray(im)
            imsave("part6/training/"+tr, im)
        for te in test:
            im = imread("cropped/"+ te)
            img = rgb2gray(im)
            imsave("part6/test/"+te, im)
        for va in vali :
            im = imread("cropped/"+ va)
            img = rgb2gray(im)
            imsave("part6/validation/"+va, im)

    
def deleteallimagepart6():
    for f in os.listdir("part6/training"):
        os.remove("part6/training/"+f)
    for f in os.listdir("part6/test"):
        os.remove("part6/test/"+f)
    for f in os.listdir("part6/validation"):
        os.remove("part6/validation/"+f)
