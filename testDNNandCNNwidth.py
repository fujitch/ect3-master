# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

dataset = pickle.load(open('dataset.pickle', 'rb'))

"""
datasetTrain = pickle.load(open("datasetTrainDNN.pickle", "rb"))
datasetTest = pickle.load(open("datasetTestDNN.pickle", "rb"))
"""

datasetDummy = []
for data in dataset:
    if data[2] == 7:
        if data[4] == 400:
            datasetDummy.append(data)
dataset = datasetDummy

random.shuffle(dataset)
datasetMatrix = np.zeros((len(dataset), dataset[0].shape[0]))
for i in range(len(dataset)):
    datasetMatrix[i, :] = dataset[i]
dataset = datasetMatrix
dataset = dataset[:300, :]

coilRMS = 7
liftRMS = 5
frequencyRMS = 400
conductivityRMS = 100
widthRMS = 0.5
depthRMS = 10

# 規格化
dataset[:, 0] /= widthRMS
dataset[:, 1] /= depthRMS
dataset[:, 2] /= coilRMS
dataset[:, 3] /= liftRMS
dataset[:, 4] /= frequencyRMS
dataset[:, 5] /= conductivityRMS
dataset[:, 6:] -= np.min(dataset[:, 6:])
dataset[:, 6:] /= np.max(dataset[:, 6:])

# 型変換
dataset = np.array(dataset, dtype=np.float32)

tf.reset_default_graph()

x1 = tf.placeholder("float", shape=[None, 93])
x2 = tf.placeholder("float", shape=[None, 4])
y_ = tf.placeholder("float", shape=[None, 1])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み処理を定義
def conv2d_pad(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# プーリング処理を定義
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# 畳み込み層1
W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
x_image = tf.reshape(x1, [-1, 31, 3, 1])
h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
# プーリング層1
h_pool1 = max_pool_2_2(h_conv1)

# 畳み込み層2
W_conv2 = weight_variable([3, 2, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
# プーリング層2
h_pool2 = max_pool_2_2(h_conv2)

# 全結合層1
W_fc1 = weight_variable([64*8, 64])
b_fc1 = bias_variable([64])
h1_flat = tf.reshape(h_pool2, [-1, 64*8])
h_fc1 = tf.nn.sigmoid(tf.matmul(h1_flat, W_fc1) + b_fc1)

# 全結合層2
W_fc2 = weight_variable([4, 32])
b_fc2 = bias_variable([32])
h2_flat = tf.reshape(x2, [-1, 4])
h_fc2 = tf.nn.sigmoid(tf.matmul(h2_flat, W_fc2) + b_fc2)

hcon = tf.concat([h_fc1, h_fc2], 1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(hcon, keep_prob)

# 全結合層3
W_fc3 = weight_variable([96, 16])
b_fc3 = bias_variable([16])
h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

h_fc2_drop = tf.nn.dropout(h_fc3, keep_prob)

# 全結合層4
W_fc4 = weight_variable([16, 1])
b_fc4 = bias_variable([1])
y_out = tf.matmul(h_fc2_drop, W_fc4) + b_fc4

each_square = tf.square(y_ - y_out)
loss = tf.reduce_mean(each_square)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()

saver = tf.train.Saver()

# sess.run(tf.initialize_all_variables())
saver.restore(sess, "./20181104DNNandCNNmodelwidth")

batch1 = dataset[:, 6:]
batch2 = dataset[:, 2:6]
output = dataset[:, :1]
out = y_out.eval(session=sess, feed_dict={x1: batch1, x2: batch2, y_: output, keep_prob: 1.0})
output = output[:300, :]
out = out[:300, :]

sess.close()
