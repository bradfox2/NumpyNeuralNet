"""Implement a basic 3 layer fully connected NN in numpy using autograd for autodifferentiation, if needed.
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

from NNN.activation_functions import sigmoid, relu
from NNN.lossfunctions import bce_loss, d_bce_loss
from NNN.layers import Layer, LinearLayer
from NNN.initializers import Initializer

import click

#np.random.seed(1)

@click.command()
@click.option('--epochs', default=1000, help='Number of epochs.')
def run(epochs):

    lr = 0.01

    cost = []

    x = np.array([[0,0],[0,100.],[100.,0],[1.,1.]])
    y = np.array([[0,1.,0,1.]]).T

    print(y)

    layer_0 = x
    layer_1 = LinearLayer(2, 4, relu, weight_initialization_function=Initializer.random_normal)
    layer_2 = LinearLayer(4, 4, relu, weight_initialization_function=Initializer.relu_uniform, num_layers = 3)
    layer_3 = LinearLayer(4, 1, sigmoid, weight_initialization_function=Initializer.sigmoid_uniform)

    cost = []

    for i in range(epochs):

        hl3 = layer_3(layer_2(layer_1(layer_0)))
        loss = np.average(bce_loss(hl3, y))
        dloss = hl3 - y

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

if __name__ == '__main__':
    run()