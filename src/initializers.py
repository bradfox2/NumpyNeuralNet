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
    @staticmethod
    def zeros(input_size, output_size, **kwargs):
        return np.zeros((input_size, output_size))

    #random normal
    @staticmethod
    def random_normal(input_size, output_size, **kwargs):
        """Gaussian.
        """
        return np.random.randn(input_size, output_size)

    #xavier
    @staticmethod
    def xavier(input_size, output_size, **kwargs):
        """In this case, assigning the weights follow a Gaussian distribution with zero mean and a finite variance is a good choice. Here, the Xavier initialization, proposed by Xavier Glorot and Yoshua Bengio, is a Gaussian distribution with zero mean and a suitable variance, which makes sure the weights are “just right”, keeping the signal in a reasonable range of values through many layers.
        """
        return 2/(input_size + output_size)

    @staticmethod
    def xavier_uniform(input_size, output_size, **kwargs):
        bound = 6**(1/2)/((input_size + output_size)**(1/2))
        return np.random.uniform(-bound,
                            bound,
                            (input_size, output_size))


    #sigmoid uniform 
    @staticmethod
    def sigmoid_uniform(input_size, output_size, **kwargs):
        bound = 4 * (6**(1/2))/((input_size + output_size)**(1/2))
        return np.random.uniform(-bound,
                            bound,
                            (input_size, output_size))

    #ReLU
    @staticmethod
    def uniform_distribution(input_size, output_size, **kwargs):
        return np.random.uniform(-1/(input_size**(1/2)),1/(input_size**(1/2)),(input_size, output_size))

    #ReLU uniform
    @staticmethod
    def relu_uniform(input_size, output_size, **kwargs):
        """Used for deep CNNs on relu layers only.
        
        Arguments:
            num_layers {[type]} -- total number of layers
        """
        print(kwargs)
        #2 layers of passed kwargs, nested dicts
        num_layers = kwargs['custom_params']['num_layers']
        bound = (6/num_layers)**(1/2)
        return np.random.uniform(-bound, bound, (input_size, output_size))
