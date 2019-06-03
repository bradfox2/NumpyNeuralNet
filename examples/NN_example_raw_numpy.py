import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def bce_loss(pred, y):
    return -(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def d_bce_loss(pred, y):
    return y - pred

lr = 0.1

x = np.array([[0,0],[0,1.],[1.,0],[1.,1.]])
y = np.array([[0,1.,0,1.]]).T

#x = np.array([[1,0], [0,1]])
#y = np.array([[1], [0]])

cost = []
np.random.seed(1)

w0 = 2 * np.random.random((2, 4)) - 1
b0 = np.zeros(4)
w1 = 2 * np.random.rand(4, 8) - 1
b1 = np.zeros(8)
w2 = np.random.rand(8, 1)
b2 = np.zeros(1)

#loop over next lines to train weights, biases
#set h0 to x so that we can line up hidden layer, summation layer, and activation layer indexes
h0 = x
s0 = (x @ w0) + b0
h1 = sigmoid(s0) #(4x4)
s1 = (h1 @ w1) + b1
h2 = sigmoid(s1) #(4x8)
s2 = h2 @ w2 + b2
h3 = sigmoid(s2) #(8x1)

loss = bce_loss(h3, y)
cost.append(np.average(loss))
#for positive class error is BCE - 1, for negative, just BCE - 0, or the same as our Y classes!
dloss = h3 - y

ds2 = d_sigmoid(s2) * dloss
#the loss for dloss/db = 1, so bias error is just the gradient of loss x 1, np sum is easy way to get the total gradients
db2 = dloss
# n x a * a x m = n x m
# fwd pass h1 x w2 = h2
dw2 = h2.T @ ds2
dh2 = ds2 @ w2.T

ds1 = d_sigmoid(s1) * dh2
db1 = dloss
dw1 = h1.T @ ds1
dh1 = ds1 @ w1.T

ds0 = d_sigmoid(s0) * dh1
db0 = np.sum(dh1, axis = 0)
dw0 = h0.T @ ds0
dh0 = ds0 @ w0.T

w2 = w2 - dw2 * lr
w1 = w1 - dw1 * lr
w0 = w0 - dw0 * lr

b2 = b2 - db2 * lr
b1 = b1 - db1 * lr
b0 = b0 - db0 * lr

print(h3)