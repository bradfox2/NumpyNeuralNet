import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

from NNN.activation_functions import sigmoid, relu
from NNN.lossfunctions import bce_loss, d_bce_loss
from NNN.layers import Layer, LinearLayer
from NNN.initializers import Initializer
from NNN.utils import Data

import collections
import click

lr = 0.01

x = np.array([[0,0,1.],[0,1.,0],[1.,0,1.],[1.,1.,1.]])
y = np.array([[0,1.,0,1.]]).T

epochs = 1000
batch_size = 2

if not batch_size:
    batch_size = len(x)

mini_batches_per_epoch = int(len(x)/batch_size) #int rounds fp down
data = Data(x,y,batch_size)

layer_1 = LinearLayer(3, 4, relu, weight_initialization_function=Initializer.random_normal)
layer_2 = LinearLayer(4, 4, relu, weight_initialization_function=Initializer.relu_uniform, num_layers = 3)
layer_3 = LinearLayer(4, 1, sigmoid, weight_initialization_function=Initializer.sigmoid_uniform)

cost = []

for i in range(epochs):
    for idx, mb in enumerate(range(mini_batches_per_epoch)):
        #print(idx)
        mb = next(data.mini_batch)
        layer_0 = mb.train
        hl3 = layer_3(layer_2(layer_1(layer_0)))
        loss = np.average(bce_loss(hl3, y))
        dloss = hl3 - mb.target

        _ = layer_1.backward_pass(layer_2.backward_pass(layer_3.backward_pass(dloss, lr), lr), lr)
        
        if i % 1000 == 0:
            print(loss)
        cost.append(loss)    

print(hl3)
print(y)
plt.plot(cost)
plt.ylabel('Loss')
plt.title('{}'.format(loss))
plt.show()