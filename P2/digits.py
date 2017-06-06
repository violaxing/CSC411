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

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set

# PART 1 code
'''
f, axarr = plt.subplots(10, 10)
for i in range(10):
    for i2 in range(10):
        axarr[i, i2].imshow(M["train"+str(i)][i2].reshape((28,28)), cmap=cm.gray)
        axarr[i,i2].axis('off')
        
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.show()
'''

def part2(x, W, b):
    ''' Computes the network in part 2, is a 784 vector
        W is a 10 x 784 matrix, where
        b i s 10 dimensional vector
    '''
    outputs = np.dot(W, x) + b
    # print outputs.shape
    return softmax(outputs)

# part 3b
def compute_grad(x, W, b, y, j, i):
    ''' Compute gradient with respect to the weight W_ji
        x is a 784 x M matrix, where M is the number of training examples
        W is a 10 x 784 matrix
        b is a 10 dimensional vector,
        y is 10 x M matrix, where M is the number of training examples
    '''
    
    return sum(((p[i, :] - y[i, :])*x[j, :].T))

def compute_grad_matrix(x, W, b, y):
    p = part2(x, W, b)
    grads = zeros((784, 10))
    for j in range(784):
        #for i in range(10):
        grads[j, :] = np.dot((p[:, :] - y[:, :]), x[j, :].T)
    return grads

def cost_function(y, p):
    ''' y is a 10 x M matrix
        p is a 10 x M matrix
    '''
    m = sum(np.multiply(y, log(p)), axis=0)
    return-1*sum(m)

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    # print tile(sum(exp(y),0), (len(y),1))
    # print exp(y)
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    print L1
    output = softmax(L1)
    return L0, L1, output
    
def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    

#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network




#########################################
NUM_TRAINING_EXAMPLES_PER_NUMBER = 500  #
#########################################
# load 100 of each
# x = M["train5"][148:169]
y = np.zeros((10, NUM_TRAINING_EXAMPLES_PER_NUMBER*10))
y_val = np.zeros((10, 200*10))
y_test = np.zeros((10, 200*10))
for i in range(10):
    if i == 0:
        x = M["train0"][0:NUM_TRAINING_EXAMPLES_PER_NUMBER]
        validation = M["train0"][NUM_TRAINING_EXAMPLES_PER_NUMBER:NUM_TRAINING_EXAMPLES_PER_NUMBER+200]
        test = M["train0"][NUM_TRAINING_EXAMPLES_PER_NUMBER + 200:NUM_TRAINING_EXAMPLES_PER_NUMBER+400]
        y[i,0:NUM_TRAINING_EXAMPLES_PER_NUMBER] = 1
        y_val[i, 0:200] = 1
        y_test[i, 0:200] = 1
    else:
        x = vstack((x, M["train"+str(i)][0:NUM_TRAINING_EXAMPLES_PER_NUMBER]))
        validation = vstack((validation, M["train"+str(i)][NUM_TRAINING_EXAMPLES_PER_NUMBER:NUM_TRAINING_EXAMPLES_PER_NUMBER+200]))
        test = vstack((test, M["train"+str(i)][NUM_TRAINING_EXAMPLES_PER_NUMBER + 200:NUM_TRAINING_EXAMPLES_PER_NUMBER+400]))
        starting = i*NUM_TRAINING_EXAMPLES_PER_NUMBER
        y[i,starting:starting+NUM_TRAINING_EXAMPLES_PER_NUMBER] = 1
        y_val[i, i*200:i*200+200] = 1
        y_test[i, i*200:i*200+200] = 1
# x = M["train5"][148:169]


x = x.T
x = x/255.0

validation = validation.T
validation = validation/255.0

test = test.T
test = test/255.0
# print W0.shape # (784, 300)
# print W1.shape # (300, 10)

# initialize the weights like so
W = np.dot(W0, W1)
'''
print x.shape # (784, 1)
print b1.shape
print W.shape
L0, L1, output = forward(x, W0, b0, W1, b1)
print output
'''

# initialize weigths like this
#get the index at which the output is the largest

# print compute_grad(x, W.T, b1, output, 2, 1)
# print cost_function(output, part2(x, W.T, b1))



# PART 4 training the network
h = 0.000000001


def grad_descent(f, df, x, y, b1, init_t, alpha):
    EPS = 1e-10   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 5000 # 30000
    itr  = 0
    while norm(t - prev_t) >  EPS and itr < max_iter:
        prev_t = t.copy()
        a = alpha*df(x, t, b1, y).T
        t -= a
        b1 -= alpha*ones((10, 1))
        if itr % 500 == 0:
            print "Iter", itr
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0,0], t[1,1], t[2,2], f(y, part2(x, t, b1))) 
            print "Gradient: ", df(x, t, b1, y), "\n"
            print "training set: ", get_performance(y, x, t, b1), "\n"
            print "validation set: ", get_performance(y_val, validation, t, b1),  "\n"
            print "test set: ", get_performance(y_test, test, t, b1),  "\n"
        itr += 1
    return t, b1

def get_performance(y, x, W, bias):
    total = float(x.shape[1])
    correct = 0
    values = part2(x, W, bias)
    for i in range(x.shape[1]): 
        if (y[argmax(values[:,i]),i] == 1):
            correct +=1
    print "performance: ", correct/total, "\n"
    return correct/total

# PART 3b checking that gradient is correct with finite differences
def check_finite_differences():
    for j in range(784):
        for i in range(10):
            a = zeros((784, 10))
            a[j, i] = h
            exact = compute_grad_matrix(x, W.T, b1, y)      
            c0 = cost_function(y, part2(x, W.T, b1))
            c1 = cost_function(y, part2(x, (W + a).T, b1))
            c2 = cost_function(y, part2(x, (W - a).T, b1))
            if ((c1 - c2)/(2 * h) != 0): 
                print i,j
                print (c1 - c2)/(2. * h)
                print exact[j, i]



# UNCOMMENT TO RUN 

# check_finite_differences()
final_weights, bias = grad_descent(cost_function, compute_grad_matrix, x, y, b1, W.T, 0.001)
#get_performance(y, x, final_weights, bias)


################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
#show heatmap for each i, should be the number corresponding to it


# PART 4, display the weights going into each of the output units

for i in range(10):
    fig = figure(i)
    ax = fig.gca()
    heatmap = ax.imshow(final_weights[i,:].reshape((28,28)), cmap = cm.coolwarm)    
    fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    savefig("figure_"+str(i)+".png")
    # show()

# PART 4 plot the learning curves
#training set
plt.plot([0,500,1000,1500, 2000, 2500, 3000, 3500, 4000, 4500], [0.3294,0.8822, 0.9906, 0.997, 0.9986, 0.9994, 0.9998, 0.9998, 1.0, 1.0], 'r', label='training')
# validation
plt.plot([0,500,1000,1500, 2000, 2500, 3000, 3500, 4000, 4500 ], [0.317,0.8245, 0.8945, 0.893, 0.8925, 0.8925, 0.892, 0.892, 0.8925, 0.8925], 'b', label='validation')
# test
plt.plot([0,500,1000,1500, 2000, 2500, 3000, 3500, 4000, 4500], [0.316,0.8145, 0.8715, 0.8675,0.8675,0.8665, 0.866, 0.8655, 0.8645, 0.862], 'g', label='test')
plt.axis([0, 4000, 0, 1.1])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()
