import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def bce_loss(pred, y):
    return -(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def d_bce_loss(pred, y):
    return y - pred

lr = 0.1

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0,0,1,1]]).T

#x = np.array([[1,0], [0,1]])
#y = np.array([[1], [0]])

cost = []
np.random.seed(1)

w0 = 2 * np.random.random((2, 4)) - 1
#b0 = np.zeros(1)
w1 = 2 * np.random.rand(4, 1) - 1
#b1 = np.zeros(4)
#w2 = np.random.rand(4, 1)
#b2 = np.zeros(1)
    
for i in range(10000):
    h0 = x
    s0 = x @ w0 #+ b0
    h1 = sigmoid(s0) #(4x4)
    s1 = h1 @ w1 #+ b1
    h2 = sigmoid(s1) #(4x4)
    #s2 = h1 @ w2 + b2
    #h2 = sigmoid(s2) #(4x1)

    loss = bce_loss(h2, y)
    cost.append(np.average(loss))
    #for positive class error is BCE - 1, for negative, just BCE - 0, or the same as our Y classes!
    dloss = h2 - y

    dh1 = d_sigmoid(s1) * dloss
    #db1 = np.sum(dloss)
    # n x a * a x m = n x m
    # fwd pass h1 x w2 = h2
    dw1 = h1.T @ dh1

    dh0 = d_sigmoid(s0) * dh1
    #db1 = np.sum(dh2)
    dw0 = h0.T @ dh0

    #dh0 = d_sigmoid(s0) * dh1
    #db0 = np.sum(dh1)
    #dw0 = x.T @ dh0

    #print(dw0, dw1, dw2)

    #w2 = w2 - dw2 * lr
    w1 = w1 - dw1 * lr
    w0 = w0 - dw0 * lr

    #b2 = b2 - db2 * lr
    #b1 = b1 - db1 * lr#
    #b0 = b0 - db0 * lr

    print(h2)