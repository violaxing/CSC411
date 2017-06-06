from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from rgb2gray import *
import cPickle
import hashlib
import random
import os
from scipy.io import loadmat


t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)


M = loadmat("mnist_all.mat")

import tensorflow as tf
f = open('subset_actors.txt', 'w')
act= ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act2 = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()  
def get_act(act):
    f = open("faces_subset.txt", 'w')
    for a in open("facescrub_actors.txt").readlines():
        f.write(a)
    for a in open("facescrub_actresses.txt").readlines():
        f.write(a)
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                

                print filename
                # get face crop coords
                face_coords = line.split("\t")[4]
                sha256 = line.split("\t")[5]
                x1 = int(face_coords.split(',')[0])
                y1 = int(face_coords.split(',')[1])
                x2 = int(face_coords.split(',')[2])
                y2 = int(face_coords.split(',')[3])
                #crop image to get face
                try:
                    im = imread("uncropped/"+filename)
                   
                    file1 = open("uncropped/"+filename).read()
                    m = hashlib.sha256()
                    m.update(file1)
                    
                    if (m.hexdigest() != sha256.strip()):
                      continue
                    
                   # print sha256
                   # print m.hexdigest()
                
                except:
                    print("Couldn't read the file")
                    continue
               
                try:
                    face_croppedim = im[y1:y2, x1:x2]
                    # grayim = rgb2gray(face_croppedim)
    #                   scaledim = imresize(face_croppedim, (32, 32))
         #              imshow(scaledim, cmap=cm.gray)
                    imsave("unscaled/"+filename, face_croppedim, cmap=cm.gray)
                    '''
                    if (i < 100):
                        imsave("rgbtraining/"+filename, scaledim, cmap=cm.gray)
                    elif (i < 130):
                        imsave("rgbval/"+filename, scaledim, cmap=cm.gray)
                    elif (i < 160):
                        imsave("rgbtest/"+filename, scaledim, cmap=cm.gray)
                        '''
                    i += 1
                
                except:
                    print("too many indices for array")
                    continue
                
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)



# SET IMG DIMENSIONS HERE
img_dim = 60


def make_rgb_sets():
    for act in act2:
        for i in range(160):
            try:
                im = imread("unscaled/"+act+str(i)+".jpg", True)
            except:
                try: 
                    im = imread("unscaled/"+act+str(i)+".png", True)
                except:
                    continue
            # face_croppedim = im[y1:y2, x1:x2]
            scaledim = imresize(im, (img_dim, img_dim))
            if (i < 100):
                imsave("rgbtraining/"+act+str(i), scaledim)
            elif (i < 130):
                imsave("rgbtest/"+act+str(i), scaledim)
            elif (i < 160):
                imsave("rgbval/"+act+str(i), scaledim)
                

def get_training_set(folder):
    dct = {}
    for act in act2:
        a_index = 0
        array = np.ones((100, img_dim*img_dim))
        for i in range(100):
            try:
                im = imread(folder+act+str(i)+".jpg", True)
            except:
                im = imread(folder+act+str(i)+".png", True)
            array[a_index,:] = im.flatten()
            a_index = a_index + 1
 #           array = array/255.0
            dct[act] = array
    return dct
    
def get_validation_set(folder):
    dct = {}
    for act in act2:
        a_index = 0
        array = np.ones((9, img_dim*img_dim))
        for i in range(9):
            try:
                im = imread(folder+act+str(i+130)+".jpg", True)
            except:
                im = imread(folder+act+str(i+130)+".png", True)
            array[a_index,:] = im.flatten()
            a_index = a_index + 1
  #          array = array/255.0
            dct[act] = array
    return dct

def get_test_set(folder):
    dct = {}
    for act in act2:
        a_index = 0
        array = np.ones((30, img_dim*img_dim))
        for i in range(30):
            try:
                im = imread(folder+act+str(i+100)+".jpg", True)
            except:
                im = imread(folder+act+str(i+100)+".png", True)
            array[a_index,:] = im.flatten()
            a_index = a_index + 1
            # array = array/255.0
            dct[act] = array
    return dct

# for rgb
make_rgb_sets()
training_dict = get_training_set("rgbtraining/")
test_dict = get_test_set("rgbtest/")
val_dict = get_validation_set("rgbval/")

def get_train_batch(M, N):
    # n = N/10
    n = N/6
    #batch_xs = zeros((0, 28*28))
    #batch_y_s = zeros( (0, 10))
    batch_xs = zeros((0, img_dim*img_dim))
    batch_y_s = zeros( (0, 6))
    # train_k =  ["train"+str(i) for i in range(6)]
    # act2 = ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell']
    train_k =  act2
    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(6): #changed range to 6 because there are 6 actors
        train_size = len(M[train_k[k]])
        idx = array(np.random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, img_dim*img_dim))
    batch_y_s = zeros( (0, 6))
    
    # test_k =  ["test"+str(i) for i in range(6)]
    test_k = act2
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s

def get_validation(M):
    batch_xs = zeros((0, img_dim*img_dim))
    batch_y_s = zeros( (0, 6))
    
    # test_k =  ["val"+str(i) for i in range(6)]
    test_k = act2
    for k in range(6):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
def get_train(M):
    # batch_xs = zeros((0, 28*28))
    # batch_y_s = zeros( (0, 10))
    batch_xs = zeros((0, img_dim*img_dim))
    batch_y_s = zeros( (0, 6))
    
    # train_k =  ["train"+str(i) for i in range(10)]
    train_k =  act2
    for k in range(6): # changed range to 6 because 6 actors
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(6)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        



x = tf.placeholder(tf.float32, [None, img_dim*img_dim])


nhid = 10
W0 = tf.Variable(tf.random_normal([img_dim*img_dim, nhid], stddev=0.01))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
'''
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = tf.Variable(snapshot["W0"])
b0 = tf.Variable(snapshot["b0"])
W1 = tf.Variable(snapshot["W1"])
b1 = tf.Variable(snapshot["b1"])
'''


layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
# layer1 = tf.nn.relu(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])



lam = 0.00000
# lam = 50
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(test_dict)
val_x, val_y = get_validation(val_dict)

itr = []
train =[]
test = []
val=[]
for i in range(5000):
  #print i  
  batch_xs, batch_ys = get_train_batch(training_dict, 50)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  
  if i % 100 == 0:
    print "i=",i
    batch_xs, batch_ys = get_train(training_dict)
    train_accr = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    test_accr = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    val_accr = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})
    itr.append(i)
    train.append(train_accr)
    test.append(test_accr)
    val.append(val_accr)
    print "Train:", train_accr
    print "Test:", test_accr
    print "Validation:", val_accr
    print "Penalty:", sess.run(decay_penalty)

    
    snapshot = {}
    snapshot["W0"] = sess.run(W0)
    snapshot["W1"] = sess.run(W1)
    snapshot["b0"] = sess.run(b0)
    snapshot["b1"] = sess.run(b1)
    cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w")) 


# get weights for chenoweth 
im = imread("rgbtest/chenoweth100.png", True)
x = im.flatten()
W0 = sess.run(W0)
b0 = sess.run(b0)
W1 = sess.run(W1)
b1 = sess.run(b1)
layer1 = np.tanh(np.dot(x, W0)+b0)
layer2 = np.dot(layer1, W1)+b1
y = softmax(layer2)


for i in range(len(layer1)):
    if layer1[i] == 1:	
        imshow(W0[:,i].reshape((60, 60)), cmap=cm.coolwarm)
        show()
# get weights for hader
im = imread("rgbtest/hader100.png", True)
x = im.flatten()
W0 = sess.run(W0)
b0 = sess.run(b0)
W1 = sess.run(W1)
b1 = sess.run(b1)
layer1 = np.tanh(np.dot(x, W0)+b0)
layer2 = np.dot(layer1, W1)+b1
y = softmax(layer2)


for i in range(len(layer1)):
    if layer1[i] == 1:	
        imshow(W0[:,i].reshape((60, 60)), cmap=cm.coolwarm)
        show()
'''
for i in range(300):
    imshow(sess.run(W0)[:,i].reshape((32, 32)), cmap=cm.coolwarm)
    show()
'''
    
plt.plot(itr, train, 'r', label='training')
plt.plot(itr, test, 'g', label='test')
plt.plot(itr, val, 'b', label='val')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.axis([0, 1400, 0, 1.1])
plt.show()
