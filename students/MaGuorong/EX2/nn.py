# coding=utf-8

import numpy as np
from image import image2onebit as it
import sys

from tensorflow.examples.tutorials.mnist import input_data
import math
import datetime
from PIL import Image
train_sum = 5000


def get_index(train_data, test_data, i):
    # 1
    #	return np.argmin(np.sqrt(np.sum(np.power(test_X[i]-train_X,2),axis=1)))
    # 2
    #	print("test_Y[i]=",test_Y[i])
    #print(test_data[i])
    return np.argmin(np.sqrt(np.sum(np.square(test_data[i] - train_data), axis=1)))


# 3	return np.argmin(np.linalg.norm(test_X[i]-train_X,axis=1))


# 1
# 读取训练与测试用的图片数据集
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# 取训练数据
train_data, train_label = mnist.train.next_batch(train_sum)

# 2
# 准备计算用的数据内存空间
min_index = 0


# 转换成numpy.array类型
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


testNum = int(input("input the number of test picture:"))
for i in range(testNum):
    testPicture = input("输入文件名:")
    testName = '/home/mgr/PycharmProjects/untitled/png/' + testPicture
    x1 = getTestPicArray(testName)

    t1 = datetime.datetime.now()
    # 3

    min_index = get_index(train_data, x1, 0)
    print('predicted: ', np.argmax((train_label[min_index])))

    # 4
    t2 = datetime.datetime.now()
    print(t2 - t1)
