import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

def sigmoid(x):
    """Squash x to between 0 and 1.
    """
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    """Derivative of sigmoid at x.
    """
    return sigmoid(x) * (1-sigmoid(x))

def relu(x):
    """Rectified linear activation.
    
    Arguments:
        x {array of numbers} 
    
    Returns:
        [array type same as input] -- 0 if x is less than 0 else x
    """
    return np.maximum(x,0)

def d_relu(x):
    """Lazy derivative of relu.
    """
    return elementwise_grad(relu)