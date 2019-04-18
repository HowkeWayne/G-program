# -*- coding: utf-8 -*-
#   Author: HowkeWayne
#     Date: 2019/4/18 - 9:54
"""
File Description...
11行神经网络
搭建一个具有两层的神经网络
"""
import numpy as np


# sigmoid function has special feature .  derivative = out*(1-out)
def nonlin(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def case_1():
    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    # layer-1 has 4nodes
    syn0 = 2 * np.random.random((3, 4)) - 1
    # layer-2 has 1node
    syn1 = 2 * np.random.random((4, 1)) - 1
    # 60000 reverse iterations / Gradient descent
    for j in range(60000):
        # Sigmoid activation function l1&l2
        l1 = 1 / (1 + np.exp(-(np.dot(x, syn0))))
        l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))
        # reverse Error
        l2_delta = (y - l2) * (l2 * (1 - l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))
        # weights updated
        syn1 += l1.T.dot(l2_delta)
        syn0 += x.T.dot(l1_delta)


if __name__ == '__main__':
    case_1()
