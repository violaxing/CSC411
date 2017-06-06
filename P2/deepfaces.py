from pylab import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import scipy.io as sio
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from PIL import Image
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf
import pickle

import os
from scipy.io import loadmat

t = int(time.time())
# t = 1454219613
print("t=", t)
random.seed(t)

act_num = 6
ndim = 100
ndimsqr = ndim ** 2 *3
alex_dim = 9600 

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.

def preprocess(imagesDir,act,size):
    data={}
    cnt  = 0
    for a in act:
        i, j, k = 0, 0, 0
        for file in os.listdir(imagesDir+a)[:size[0]]:
            if i == 0:
                data['train' + str(cnt)]=[]
            if not os.path.isdir(file) and  a in file:
                #lena = Image.open(imagesDir+a+'/'+file)
                #new_img = lena.convert('RGB')
                #new_img.save('new_img/'+a+'/'+file)
                data['train'+str(cnt)].append(mpimg.imread('new_img/'+a+'/'+file).reshape(ndimsqr,))
            i += 1

        for file in os.listdir(imagesDir+a)[size[0]:size[0]+size[1]]:
            if j == 0:
                data['valid' + str(cnt)]=[]
            if not os.path.isdir(file) and  a in file:
                #lena = Image.open(imagesDir + a + '/' + file)
                #new_img = lena.convert('RGB')
                #new_img.save('new_img/' + a + '/' + file)
                data['valid'+str(cnt)].append(mpimg.imread('new_img/'+a+'/'+file).reshape(ndimsqr,))
            j += 1
        for file in os.listdir(imagesDir+a)[-size[2]:]:
            if k == 0:
                data['test' + str(cnt)]=[]
            if not os.path.isdir(file) and  a in file:
                # lena = Image.open(imagesDir + a + '/' + file)
                # new_img = lena.convert('RGB')
                # new_img.save('new_img/' + a + '/' + file)
                data['test'+str(cnt)].append(mpimg.imread('new_img/'+a+'/'+file).reshape(ndimsqr,))
            k += 1
        cnt += 1

    sio.savemat('savemat2.mat',data)





def get_train_batch(M, N):
    n = N / act_num
    batch_xs = np.zeros((0,) + M['train0'][0].shape)
    batch_y_s = np.zeros((0, act_num))
    train_k = ["train" + str(i) for i in range(act_num)]
    train_size = len(M[train_k[0]])

    for k in range(act_num):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx]) / 255.)))
        one_hot = np.zeros(act_num)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (n, 1))))
    return batch_xs, batch_y_s


def get_test(M):
    batch_xs = np.zeros((0,) + M['test0'][0].shape)
    batch_y_s = np.zeros((0, act_num))

    test_k = ["test" + str(i) for i in range(act_num)]
    for k in range(act_num):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:]) / 255.)))
        one_hot = np.zeros(act_num)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[test_k[k]]), 1))))
    return batch_xs, batch_y_s


def get_valid(M):
    batch_xs = np.zeros((0,) + M['valid0'][0].shape)
    batch_y_s = np.zeros((0, act_num))

    valid_k = ["valid" + str(i) for i in range(act_num)]
    for k in range(act_num):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:]) / 255.)))
        one_hot = np.zeros(act_num)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[valid_k[k]]), 1))))
    return batch_xs, batch_y_s


def get_train(M):
    batch_xs = np.zeros((0,) + M['train0'][0].shape)
    batch_y_s = np.zeros((0, act_num))

    train_k = ["train" + str(i) for i in range(act_num)]
    for k in range(act_num):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:]) / 255.)))
        one_hot = np.zeros(act_num)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[train_k[k]]), 1))))
    return batch_xs, batch_y_s

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]

    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

# Extracting the values of the activations
def get_activations(data):
    #x = tf.placeholder(tf.float32, [1, ndimsqr]) 
    x = tf.placeholder(tf.float32, [1, 100, 100,3 ]) 
    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    print(conv1W)
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    conv_sets = [np.empty((1, alex_dim)),
                 np.empty((1, alex_dim)),
                 np.empty((1, alex_dim))]

    data_names = ['test', 'valid', 'train']
    data_sets = [get_test(data)[0], get_valid(data)[0], get_train(data)[0]]
    for i in range(3):
        set_size = data_sets[i].shape[0]
        data_set = data_sets[i].reshape(set_size,100,100,3)
        for j in range(set_size):
            im = sess.run(conv4, feed_dict={x: array([data_set[j]])})
            im = np.reshape(im, (1, alex_dim))
            conv_sets[i] = np.concatenate((conv_sets[i], im))

    conv_sets[0] = conv_sets[0][1:]
    conv_sets[1] = conv_sets[1][1:]
    conv_sets[2] = conv_sets[2][1:]

    activations = {}
    for i in range(3):
        set_size = data_sets[i].shape[0] / 6
        for j in range(6):
            idx = set_size * j
            activations[data_names[i] + str(j)] = conv_sets[i][idx:(idx + set_size)]

    return activations


def alexNet(itera=5000): #part10
    activations = get_activations(Q) # new rgb
    x = tf.placeholder(tf.float32, [None, alex_dim])

    W1 = tf.Variable(tf.random_normal([alex_dim, act_num], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([act_num], stddev=0.01))
    layer1 = x
    layer2 = tf.matmul(layer1, W1) + b1

    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, act_num])

    lam = 0.1

    decay_penalty = lam * tf.reduce_sum(lam * tf.reduce_sum(tf.square(W1)))
    NLL = -tf.reduce_sum(y_ * tf.log(y) + decay_penalty)

    train_step = tf.train.AdamOptimizer(5e-4).minimize(NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_x, test_y = get_test(activations)
    valid_x, valid_y = get_valid(activations)
    train_x, train_y = get_train(activations)
    train_accuracy = [];valid_accuracy = [];test_accuracy = []

    i = 0
    while i < itera:
        i += 1
        batch_xs, batch_ys = get_train_batch(activations, 600)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            print("i=", i)
            v_acc = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            print("Valid:", v_acc)
            valid_accuracy.append(v_acc)
            pred = sess.run(y, feed_dict={x: valid_x, y_: valid_y})
            t_acc = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            print("Test:", t_acc)
            test_accuracy.append(t_acc)

            tr_acc = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            print("Train:", tr_acc)
            train_accuracy.append(tr_acc)

            print("Penalty:", sess.run(decay_penalty))

    print(np.argmax(pred, 1))

    snapshot = {}
    snapshot["W1"] = sess.run(W1)
    snapshot["b1"] = sess.run(b1)
    pickle.dump(snapshot, open("part10_weights.pkl", "wb"))

    plt.plot(train_accuracy, "r", label="train")
    plt.plot(valid_accuracy, "g", label="valid")
    plt.plot(test_accuracy, "b", label="test")
    title = "Learning Curves - part 10"
    plt.title(title)
    plt.legend(loc="best")
    plt.savefig(title)


if __name__ == "__main__":
    preprocess('new_img/', ['drescher', 'ferrera', 'chenoweth', 'baldwin', 'hader', 'carell'], [110, 20, 30])
    Q = loadmat("savemat2.mat") # new rgb
    alexNet()

