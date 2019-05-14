import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

class Initializer(object):
    """Weight initialiation methods.
    
    Arguments:
        input_size {int} -- Num units inbound.
        output_size {int} -- Num units outbound.
    Returns:
        [array] -- array with floats init'd per selected method.
    """
    
    #zeros
    @classmethod
    def zeros(input_size, output_size):
        return np.zeros((input_size, output_size))

    #random normal
    @classmethod
    def random_normal(input_size, output_size):
        """Gaussian.
        """
        return np.random.randn(input_size, output_size)

    #xavier
    @classmethod
    def xavier(input_size, output_size):
        """In this case, assigning the weights follow a Gaussian distribution with zero mean and a finite variance is a good choice. Here, the Xavier initialization, proposed by Xavier Glorot and Yoshua Bengio, is a Gaussian distribution with zero mean and a suitable variance, which makes sure the weights are “just right”, keeping the signal in a reasonable range of values through many layers.
        """
        return 2/(input_size + output_size)

    @classmethod
    def xavier_uniform(input_size, output_size):
        bound = 6**(1/2)/((input_size + output_size)**(1/2))
        np.random.uniform(-bound,
                            bound,
                            (input_size, output_size))


    #sigmoid uniform 
    @classmethod
    def sigmoid_uniform(input_size, output_size):
        bound = 4 * (6**(1/2))/((input_size + output_size)**(1/2))
        np.random.uniform(-bound,
                            bound,
                            (input_size, output_size))

    #ReLU
    @classmethod
    def uniform_distribution(input_size, output_size):
        return np.random.uniform(-1/(input_size**(1/2)),1/(input_size**(1/2)),(input_size, output_size))

    #ReLU uniform
    @classmethod
    def relu_uniform(input_size, output_size, num_layers):
        """Used for deep CNNs on relu layers only.
        
        Arguments:
            num_layers {[type]} -- total number of layers
        """
        bound = (6/num_layers)**(1/2)
        return np.random.uniform(-bound, bound, (input_size, output_size))
