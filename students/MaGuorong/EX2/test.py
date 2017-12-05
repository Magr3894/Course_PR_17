# -*- coding: utf-8 -*-
import tensorflow as tf
from input_data import read_data_sets
import time
from PIL import Image
import numpy as np

start = time.clock();
samples = read_data_sets('/home/mgr/PycharmProjects/untitled/MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
# 创建一个会话

x = tf.placeholder(tf.float32, shape=[None, 784])
# 因为mnist数据集为28*28=784
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 表示输入是一个[?,5]的矩阵。那么什么情况下会这么用呢?就是需要输入一批[1,5]的数据的时候。
# 比如我有一批共10个数据，那我可以表示成[10,5]的矩阵。如果是一批5个，那就是[5,5]的矩阵。tensorflow会自动进行批处理
'''initial weight variable'''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 产生一个正态分布
    return tf.Variable(initial)


# 偏置向量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积函数
# canshu
# input：待卷积的数据。格式要求为一个张量，[batch, in_height, in_width, in_channels].
# 分别表示 批次数，图像高度，宽度，输入通道数。
# filter： 卷积核。格式要求为[filter_height, filter_width, in_channels, out_channels].
# 分别表示 卷积核的高度，宽度，输入通道数，输出通道数。
# strides :一个长为4的list. 表示每次卷积以后卷积窗口在input中滑动的距离
# padding ：有SAME和VALID两种选项，表示是否要保留图像边上那一圈不完全卷积的部分。如果是SAME，则保留
# use_cudnn_on_gpu ：是否使用cudnn加速。默认是True
def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


# 池化函数
# value: 一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
# ksize: 长为4的list,表示池化窗口的尺寸
# strides: 池化窗口的滑动值，与conv2d中的一样
# padding: 与conv2d中用法一样。

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 第一层卷积:由一个卷积和一个池化组成
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, padding="SAME") + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, padding="VALID") + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层1 加入一个具有120个神经元的全连接层
W_fc1 = weight_variable([5 * 5 * 16, 120])
b_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 密集连接层2
W_fc2 = weight_variable([120, 84])
b_fc2 = bias_variable([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
# Dropout
# 为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
keep_prob = tf.placeholder("float")
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)  # 即随机安排一些cell输出值为0，可以防止过拟合

# softmax
W_softmax = weight_variable([84, 10])
b_softmax = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_softmax) + b_softmax)

# 训练与评估
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # reduce_num取和降维
# 交叉熵可在神经网络(机器学习)中作为损失函数，p表示真实标记的分布，q则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量p与q的相似性。
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()  # defaults to saving all variables
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = samples.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

        print("step %d,training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g" % accuracy.eval(feed_dict={x: samples.test.images, y_: samples.test.labels, keep_prob: 1.0}))

end = time.clock()
print("running time is: %d" % (end - start))
saver.save(sess, '/home/mgr/PycharmProjects/untitled/model.ckpt')  # 保存模型参数，注意把这里改为自己的路径


def getTestPicArray(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)

    im_arr = np.array(out.convert('L'))

    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if (num255 > num0):
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if (im_arr[x][y] < threshold):  im_arr[x][y] = 0
                # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
                # else : im_arr[x][y] = 255
                # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

                #   out = Image.fromarray(np.uint8(im_arr))
                #    out.save(filename.split('/')[0] + '/28pix/' + filename.split('/')[1])
    # print im_arr
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


def testMyPicture():
    testNum = int(input("input the number of test picture:"))
    for i in range(testNum):
        testPicture = input("输入文件名:")
        testName = '/home/mgr/PycharmProjects/untitled/png/' + testPicture
        oneTestx = getTestPicArray(testName)
        ans = tf.argmax(y_conv, 1)
        print("The prediction answer is:")
        ans1 = sess.run(ans, feed_dict={x: oneTestx, keep_prob: 1})
        print(ans1[0])


# save_path = "form/cnn.ckpt"

# restore()
testMyPicture()
sess.close()
