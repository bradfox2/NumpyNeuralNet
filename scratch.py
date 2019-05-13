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
y = np.array([[0,1,0,1]]).T

#x = np.array([[1,0], [0,1]])
#y = np.array([[1], [0]])

cost = []
np.random.seed(1)

class Layer(object):
    ''' Base class for a layer.'''
    def __init__(self, inputs, input_size, output_size):
        self.inputs = inputs
        self.input_size = input_size
        self.output_size = output_size
    
    def forward_pass(self):
        ''' Forward prop the input signal.'''
        raise NotImplementedError
    
    def backward_pass(self):
        '''Backward prop the error gradient.'''
        raise NotImplementedError

class LinearLayer(Layer):
    def __init__(self, activation_function, d_activation_function, input_size, output_size):
        Layer.activation_function = activation_function
        Layer.input_size = input_size
        Layer.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.d_activation_function = d_activation_function
        self.s = None
        self.h = None

    def forward_pass(self, inputs):
        Layer.inputs = inputs
        self.s = inputs @ self.weights + self.bias
        if self.activation_function:
            self.h = self.activation_function(self.s)
        else:
            self.h = self.s
        return self.h

    def backward_pass(self, grad):
        #calculate the gradient 
        if self.activation_function:
            dh = self.d_activation_function(self.s) * grad
        else:
             dh = self.s * grad
        db = dh
        dw = self.h.T @ dh
        #update layer parameters to be more accurate!
        self.weights -= dw * lr
        self.bias -= db * lr
        return dh


w0 = 2 * np.random.random((2, 4)) - 1
b0 = np.zeros(4)
w1 = 2 * np.random.rand(4, 4) - 1
b1 = np.zeros(4)
w2 = np.random.rand(4, 1)
b2 = np.zeros(1)

#for i in range(10000):
#set h0 to x so that we can line up hidden layer, summation layer, and activation layer indexes
h0 = x
s0 = (x @ w0) + b0
h1 = sigmoid(s0) #(4x4)
s1 = (h1 @ w1) + b1
h2 = sigmoid(s1) #(4x4)
s2 = h2 @ w2 + b2
h3 = sigmoid(s2) #(4x1)


loss = bce_loss(h3, y)s
cost.append(np.average(loss))
#for positive class error is BCE - 1, for negative, just BCE - 0, or the same as our Y classes!
dloss = h3 - y

dh2 = d_sigmoid(s2) * dloss

#the loss for dloss/db = 1, so bias error is just the gradient of loss x 1, np sum is easy way to get the total gradients
db2 = dloss
# n x a * a x m = n x m
# fwd pass h1 x w2 = h2
dw2 = h2.T @ dh2

dh1 = d_sigmoid(s1) * dh2
db1 = dloss
dw1 = h1.T @ dh1

dh0 = d_sigmoid(s0) * dh1
db0 = np.sum(dh1, axis = 0)
dw0 = h0.T @ dh0

#dh0 = d_sigmoid(s0) * dh1
#db0 = np.sum(dh1)
#dw0 = x.T @ dh0

#print(dw0, dw1, dw2)

w2 = w2 - dw2 * lr
w1 = w1 - dw1 * lr
w0 = w0 - dw0 * lr

b2 = b2 - db2 * lr
b1 = b1 - db1 * lr
b0 = b0 - db0 * lr


print(h3)
