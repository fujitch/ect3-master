import pickle
import tensorflow as tf
import numpy as np
import random

batch_size = 100
training_epochs = 100000
display_epochs = 100

datasetTrain = pickle.load(open("datasetTrainDNN.pickle", "rb"))
datasetTest = pickle.load(open("datasetTestDNN.pickle", "rb"))

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

sess.run(tf.initialize_all_variables())
# saver.restore(sess, "./20181104DNNmodelwidth")
for i in range(training_epochs):
    for k in range(0, len(datasetTrain), batch_size):
        batch1 = datasetTrain[k:k+batch_size, 6:]
        batch2 = datasetTrain[k:k+batch_size, 2:6]
        output = datasetTrain[k:k+batch_size, :1]
        train_step.run(session=sess, feed_dict={x1: batch1, x2: batch2, y_: output, keep_prob: 0.5})
    
    if i%display_epochs == 0:
        batch1 = datasetTrain[:, 6:]
        batch2 = datasetTrain[:, 2:6]
        output = datasetTrain[:, :1]
        train_loss = loss.eval(session=sess, feed_dict={x1: batch1, x2: batch2, y_: output, keep_prob: 1.0})

        batch1 = datasetTest[:, 6:]
        batch2 = datasetTest[:, 2:6]
        output = datasetTest[:, :1]
        test_loss = loss.eval(session=sess, feed_dict={x1: batch1, x2: batch2, y_: output, keep_prob: 1.0})

        print(str(i) + "epochs_finished!")
        print("train_loss===" + str(train_loss))
        print("test_loss===" + str(test_loss))
    if i%1000 == 0:
        saver.save(sess, "./20181104DNNandCNNmodelwidth")
sess.close()