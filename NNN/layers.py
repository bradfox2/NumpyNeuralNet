import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad, grad

from NNN.activation_functions import sigmoid
from NNN.initializers import Initializer

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
    
    def __call__(self, input):
        return self.forward_pass(input)

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
    def __init__(self,
        input_size, 
        output_size, 
        activation_function = sigmoid, 
        d_activation_function = None, 
        weight_initialization_function = Initializer.random_normal,
        **kwargs):
        super().__init__(None, input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = weight_initialization_function(input_size, output_size, custom_params = kwargs)
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
       return "Linear Layer of size ({}, {})".format(self.input_size, self.output_size)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.s = inputs @ self.weights + self.bias
        if self.activation_function:
            self.h = self.activation_function(self.s)
        else:
            self.h = self.s
        return self.h

    def backward_pass(self, grad, lr):
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

class BatchNorm(Layer):
    """Batch normalization layer.
    
    Batch Normalization is a technique to provide any layer in a Neural Network with inputs 
    that are zero mean/unit variance.
    """
    def __init__(self, input_size, epsilon=0.001, scale=1, shift=0):
        super().__init__(None, input_size, output_size=input_size)
        self.epsilon = epsilon
        self.inputs_mean = None
        self.inputs_variance = None
        self.normalized_input = None
        # output = gamma * normalized_input + beta
        self.shift = shift #beta, learnable, controls shifting the mean up or down
        self.scale = scale #gamma, learnable, controls how much scaling is applied
        self.output = None

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.inputs_mean = BatchNorm.mb_mean(inputs)
        self.inputs_variance = BatchNorm.mb_variance(inputs, self.inputs_mean)
        self.normalized_input = BatchNorm.normalize_input(inputs, self.inputs_mean, 
                                                          self.inputs_variance, self.epsilon)
        self.output = BatchNorm.scale_and_shift(self.normalized_input, self.scale, self.shift)
        return self.output

    def backward_pass(self, grad, lr):
        # shift 
        grad_shift = grad #same as bias
        self.shift = self.shift - grad_shift

        # scale
        grad_scale = self.normalized_input.T * grad
        self.scale = self.scale - grad_scale
        
        # x
        d_normalize_input = elementwise_grad(BatchNorm.normalize_input)
        grad_normalize = d_normalize_input(self.inputs, self.inputs_mean, self.inputs_variance, self.epsilon) * grad_scale

        d_variance = elementwise_grad(BatchNorm.mb_variance)
        grad_var = d_variance(self.inputs, self.inputs_mean) * grad_normalize
        
        dmean = elementwise_grad(BatchNorm.mb_mean)
        grad_mean = dmean(self.inputs) * grad_normalize

        return grad_var + grad_mean

    def __repr__(self):
        return str(self.scale)
    
    @staticmethod
    def mb_mean(inputs):
        '''Calulate mean of the mini batch'''
        return np.mean(inputs, axis=0)

    @staticmethod
    def mb_variance(inputs, mb_mean):
        '''Calculate variance of mini batch'''
        #same as np.var(inputs)
        return np.mean((inputs - mb_mean)**2, axis=0)

    @staticmethod
    def normalize_input(inputs, inputs_mean, inputs_variance, epsilon):
       return (inputs - inputs_mean)/((inputs_variance**2 + epsilon)**(1/2))

    @staticmethod
    def scale_and_shift(inputs, gamma, beta):
        return inputs * gamma + beta
    
if __name__ == "__main__":
    # i = np.random.randn(2,2)
    # #print(np.mean(i,axis=0))
    # bn = BatchNorm(i,2,2)
    # print(bn.forward_pass(i))
    # grad = np.array([[-1,1],[-1,1]])
    # print(bn.backward_pass(grad, 1))
    pass