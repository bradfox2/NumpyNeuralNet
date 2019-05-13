"""Implement a basic 3 layer fully connected NN in numpy using autograd for autodifferentiation, if needed.
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

from src.activation_functions import sigmoid, relu
from src.lossfunctions import bce_loss, d_bce_loss
from src.layers import Layer, LinearLayer

np.random.seed(1)

lr = 0.01

cost = []

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0,1,0,1]]).T

layer_0 = x
layer_1 = LinearLayer(2, 4, relu)
layer_2 = LinearLayer(4, 4, relu)
layer_3 = LinearLayer(4, 1, sigmoid)

cost = []

for i in range(20000):

    hl3 = layer_3(layer_2(layer_1(layer_0)))
    loss = np.average(bce_loss(hl3, y))
    dloss = hl3 - y

    _ = layer_1.backward_pass(layer_2.backward_pass(layer_3.backward_pass(dloss, lr), lr), lr)
    
    if i % 1000 == 0:
        print(loss)
    cost.append(loss)    

plt.plot(cost)
plt.ylabel('Loss')
plt.show()
