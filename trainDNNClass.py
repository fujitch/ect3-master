# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

batch_size = 100
training_epochs = 100000
display_epochs = 100

datasetTrain = pickle.load(open("datasetTrainDNN.pickle", "rb"))
datasetTest = pickle.load(open("datasetTestDNN.pickle", "rb"))

# 分類の候補を作成
batchList = np.zeros((3, 3))
batchList = np.array(batchList, dtype=np.int32)
batchList[0, 0] = 1
batchList[1, 1] = 1
batchList[2, 2] = 1

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, 97])
y_ = tf.placeholder("float", shape=[None, 3])

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
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([128, 3])
b_fc3 = bias_variable([3])
y_out = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
# saver.restore(sess, "./20181104DNNmodelwidth")
for i in range(training_epochs):
    for k in range(0, len(datasetTrain), batch_size):
        batch = datasetTrain[k:k+batch_size, 2:]
        output = datasetTrain[k:k+batch_size, :1]
        outputClass = np.zeros((batch_size, 3))
        outputClass = np.array(outputClass, dtype=np.int32)
        for l in range(batch_size):
            if output[l] < 0.01:
                outputClass[l, 0] = 1
            elif output[l] > 0.05:
                outputClass[l, 2] = 1
            else:
                outputClass[l, 1] = 1
        train_step.run(session=sess, feed_dict={x: batch, y_: outputClass, keep_prob: 0.5})
    
    if i%display_epochs == 0:
        batch = datasetTrain[:, 2:]
        output = datasetTrain[:, :1]
        outputClass = np.zeros((output.shape[0], 3))
        outputClass = np.array(outputClass, dtype=np.int32)
        for l in range(output.shape[0]):
            if output[l] < 0.01:
                outputClass[l, 0] = 1
            elif output[l] > 0.05:
                outputClass[l, 2] = 1
            else:
                outputClass[l, 1] = 1
        train_acc = accuracy.eval(session=sess, feed_dict={x: batch, y_: outputClass, keep_prob: 1.0})
        batch = datasetTest[:, 2:]
        output = datasetTest[:, :1]
        outputClass = np.zeros((output.shape[0], 3))
        outputClass = np.array(outputClass, dtype=np.int32)
        for l in range(output.shape[0]):
            if output[l] < 0.01:
                outputClass[l, 0] = 1
            elif output[l] > 0.05:
                outputClass[l, 2] = 1
            else:
                outputClass[l, 1] = 1
        test_acc = accuracy.eval(session=sess, feed_dict={x: batch, y_: outputClass, keep_prob: 1.0})
        print(str(i) + "epochs_finished!")
        print("train_acc===" + str(train_acc))
        print("test_acc===" + str(test_acc))
    if i%1000 == 0:
        saver.save(sess, "./20181104DNNmodelwidth_class")
sess.close()
