import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

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
    
    def __repr__(self):
        raise NotImplementedError

class LinearLayer(Layer):
    """Implements a basic fully connected layer with activation function. 
    
    Parameters:
        input_size {int} -- number of columns of input data, number of input units/neurons
        output_size {int} -- number of columns of output data, number of output units/neurons
        activation_function {func} -- Passed function to squash the weighted sum outputs.
        d_activation_function {func} -- Passed function that is the derivative of the activation function.  
            Function accepts the same parameter as activation_function.  Pass None and the 
            derivative of the activation function will be automatically calculated.

    Returns:
        array -- Activated output for each unit, number of units specified per output_size.
    """
    def __init__(self, input_size, output_size, activation_function, d_activation_function = None):
        super().__init__(None, input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros(output_size)
        if not d_activation_function:
            self.d_activation_function = elementwise_grad(activation_function)
        else:
            self.d_activation_function = d_activation_function
        self.activation_function = activation_function
        self.s = None
        self.h = None

    def __call__(self, input):
        return self.forward_pass(input)

    def __repr__(self):
       return f"Linear Layer of size ({self.input_size, self.output_size})"

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.s = inputs @ self.weights + self.bias
        if self.activation_function:
            self.h = self.activation_function(self.s)
        else:
            self.h = self.s
        return self.h

    def backward_pass(self, grad):
        #calculate the gradient 
        if self.activation_function and self.d_activation_function:
            dh = self.d_activation_function(self.s) * grad
        else:
             dh = self.s * grad
        db = grad
        dw = self.inputs.T @ dh
        #update layer parameters to be more accurate!
        self.weights = self.weights - dw * lr
        self.bias = self.bias - db * lr
        return dh

layer_0 = x
layer_1 = LinearLayer(2, 4, sigmoid)
layer_2 = LinearLayer(4, 4, sigmoid)
layer_3 = LinearLayer(4, 1, sigmoid)

cost = []

for i in range(1000):

    hl3 = layer_3(layer_2(layer_1(layer_0)))
    loss = np.average(bce_loss(hl3, y))
    dloss = hl3 - y

    _ = layer_1.backward_pass(layer_2.backward_pass(layer_3.backward_pass(dloss)))
    
    if i % 1000 == 0:
        print(loss)
    cost.append(loss)    

plt.plot(cost)
plt.ylabel('Loss')
plt.show()


