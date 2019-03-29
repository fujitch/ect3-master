# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

datasetTrain = pickle.load(open("datasetTrainDNN.pickle", "rb"))
datasetTest = pickle.load(open("datasetTestDNN.pickle", "rb"))

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, 97])
y_ = tf.placeholder("float", shape=[None, 1])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W_fc1 = weight_variable([97, 1024])
b_fc1 = bias_variable([1024])
h_flat = tf.reshape(x, [-1, 97])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([128, 16])
b_fc3 = bias_variable([16])
h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([16, 1])
b_fc4 = bias_variable([1])
y_out = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

each_square = tf.square(y_ - y_out)
loss = tf.reduce_mean(each_square)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess = tf.Session()

saver = tf.train.Saver()

# sess.run(tf.initialize_all_variables())
saver.restore(sess, "./20181104DNNmodelwidth")

batch = datasetTrain[:, 2:]
output = datasetTrain[:, :1]
out = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
output = output[:300, :]
out = out[:300, :]

sess.close()
