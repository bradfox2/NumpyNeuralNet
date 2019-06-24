'''Example to train a 3 layer network on Iris dataset.  Results heavily dependent on initialization.'''

import collections
import copy

import random
import autograd.numpy as np
import click
import matplotlib.pyplot as plt
from autograd import elementwise_grad, grad
from sklearn import datasets
from sklearn.metrics import roc_auc_score

from NNN.activation_functions import relu, sigmoid, softmax
from NNN.initializers import Initializer
from NNN.layers import Layer, LinearLayer
from NNN.lossfunctions import (bce_loss, cross_entropy_loss, d_bce_loss, d_cross_entropy_loss)
from NNN.optimizers import momentum, vanilla
from NNN.utils import Data

# import some data to play with
iris = datasets.load_iris()
x = iris.data
y = iris.target
#combined = list(zip(iris.data,iris.target))
#random.shuffle(combined)
#x[:], y[:] = zip(*combined)

optimizer = vanilla

def one_hot_encode_integer_sequence(y):
    ohe = np.zeros([len(y),len(np.unique(y))],dtype = np.int8)
    ohe[np.arange(len(y)), y] = 1
    return ohe

ohe = one_hot_encode_integer_sequence(y)
print(ohe)
ohe_ = copy.deepcopy(ohe)
lr = .0001

epochs = 1000
batch_size = 15
if len(x)%batch_size != 0:
    raise ValueError('batch not evenly divisible by batch size')

if not batch_size:
    batch_size = len(x)

mini_batches_per_epoch = int(len(x)/batch_size) #int rounds fp down
data = Data(x, copy.deepcopy(ohe), batch_size)

layer_1 = LinearLayer(4, 30, relu, weight_initialization_function=Initializer.relu_uniform, num_layers=3, optimizer=optimizer)
layer_2 = LinearLayer(30, 30, relu, weight_initialization_function=Initializer.relu_uniform, num_layers=3, optimizer=optimizer)
layer_3 = LinearLayer(30, 3, softmax, weight_initialization_function=Initializer.sigmoid_uniform, optimizer=optimizer)

cost = []

for i in range(epochs):
    for idx, mb in enumerate(range(mini_batches_per_epoch)):
        mb = next(data.mini_batch)
        layer_0 = mb.train
        hl3 = layer_3(layer_2(layer_1(layer_0)))
        loss = np.average(cross_entropy_loss(hl3, mb.target))
        dloss = d_cross_entropy_loss(hl3, mb.target)
        _ = layer_1.backward_pass(layer_2.backward_pass(layer_3.backward_pass(dloss, lr), lr), lr)
        
    if i % 100 == 0:
        print(loss)
    cost.append(loss)    

preds = layer_3(layer_2(layer_1(x)))
print(preds)

#roc
p  = []
for i in preds:
    p.append(np.argmax(i))
p = one_hot_encode_integer_sequence(p)
print(roc_auc_score(ohe_, p))

plt.plot(cost)
plt.ylabel('Loss')
plt.title('{}'.format(loss))
plt.show()
