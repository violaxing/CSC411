from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *
import os
import matplotlib.pyplot as plt
import sys


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()

env = gym.make('CartPole-v0')

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 0.00001
gamma = 0.99



weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

hw_init = weights_init
hb_init = relu_init
mw_init = weights_init
mb_init = relu_init
sw_init = weights_init
sb_init = relu_init

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.int32, shape=(None, output_units), name='y')



layer = fully_connected(
    inputs=x,
    num_outputs=output_units,
    activation_fn=None,
    weights_initializer=hw_init,
    weights_regularizer=None,
    biases_initializer=hb_init,
    scope='layer')

all_vars = tf.global_variables()
output = softmax(layer, scope="output")

pi = tf.contrib.distributions.Bernoulli(p=output, name='pi')
pi_sample = pi.sample()
log_pi = pi.log_prob(y, name='log_pi')

Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=50
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

if not os.path.exists("weight.txt"):
    f = open("weight.txt", "w")
    writable = True
else:
    writable = False
steps = []

recorded = []
avgsteps = []
weight_store = []
track_returns = []
for ep in range(7000):
    obs = env.reset()
    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:

        ep_states.append(obs)
        env.render()

        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action[0])
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    if not args.load_model:

        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
       
        _ = sess.run([train_op],
                    feed_dict={x:np.array(ep_states),
                                y:np.array(ep_actions),
                                Returns:returns })
    steps.append(t)
    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    with tf.variable_scope("layer", reuse=True):
            	a = sess.run(tf.get_variable("weights"))
    if ep % 50 == 0:
        if len(steps) <= MEMORY:
            ab = np.average(steps)
        else: 
            ab = np.average(steps[-MEMORY:])
        print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    
        print("Mean return over the last {} episodes is {}".format(MEMORY, ab))

        if writable:
            f.write("Weight at episode "+str(ep)+":\n")
            f.write(np.array_str(a))
            f.write("\n\n")
            recorded.append(ep)
            avgsteps.append(mean_return)

if writable:
    f.close()

if not os.path.exists("avgsteps.txt"):
    np.savetxt("avgsteps.txt", np.array(avgsteps))

if not os.path.exists("recorded.txt"):
    np.savetxt("recorded.txt", np.array(recorded))
    
if not os.path.exists("part3a.png"):
    y_axis = np.loadtxt("avgsteps.txt")
    x_axis = np.loadtxt("recorded.txt")
    plt.plot(x_axis, y_axis)
    plt.xlabel('Ep')
    plt.ylabel('avg step')
    plt.savefig("part3.png")
