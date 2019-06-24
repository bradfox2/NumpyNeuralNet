import autograd.numpy as np
import click
import matplotlib.pyplot as plt
import numpy as np
from autograd import elementwise_grad, grad
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from NNN.activation_functions import relu, sigmoid
from NNN.initializers import Initializer
from NNN.layers import Layer, LinearLayer
from NNN.lossfunctions import bce_loss, d_bce_loss

# import some data to play with
iris = datasets.load_iris()
x = iris.data  # we only take the first two features.
y = iris.target

#one hot encode
onehot_encoder = OneHotEncoder(sparse=False)
y_enc = y.reshape(len(y), 1)
y = onehot_encoder.fit_transform(y_enc)

@click.command()
@click.option('--epochs', default=1000, help='Number of epochs.')
def run(epochs):

    lr = 0.01

    cost = []

    #x = np.array([[0,0],[0,100.],[100.,0],[1.,1.]])
    #y = np.array([[0,1.,0,1.]]).T

    layer_0 = x
    layer_1 = LinearLayer(4, 150, relu, weight_initialization_function=Initializer.random_normal)
    layer_2 = LinearLayer(150, 150, relu, weight_initialization_function=Initializer.relu_uniform, num_layers = 3)
    layer_3 = LinearLayer(150, 3, sigmoid, weight_initialization_function=Initializer.sigmoid_uniform)

    cost = []

    for i in range(epochs):

        hl3 = layer_3(layer_2(layer_1(layer_0)))
        loss = np.average(bce_loss(hl3, y))
        dloss = hl3 - y
        #print(dloss)
        #print(hl3)

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
