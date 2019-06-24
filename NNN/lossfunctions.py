"""Module containing loss functions and their derivatives.
"""
import numpy as np

def bce_loss(pred, y):
    """Binary Cross Entropy Loss. Use with a sigmoid final activation.
    
    Arguments:
        pred {[array of nums]} -- Predictions
        y {[array of nums]} -- Targets/Actuals    
    
    Returns:
        Array with types same as input -- Loss value 
    """
    pred[pred == 0] = 1e-15
    y[y == 0] = 1e-15
    pred[pred == 1] = 0.999999999999
    #y[y == 1] = 0.9999999999999
    
    return -(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def d_bce_loss(pred, y):
    """Derivative of Binary Cross Entropy loss taken at pred.  
    
    Arguments:
        See bce_loss.

    Returns:
        See bce_loss.
    """
    return pred - y

def cross_entropy_loss(pred, y):
    """Cross Entropy loss, typically used with a softmax final activation.
    
    Arguments:
        pred {[type]} -- [description]
        y {[type]} -- [description]
    """
    #pred[pred == 0] = 1e-15
    #y[y == 0] = 1e-15
    #pred[pred == 1] = 0.999999999999
    #y[y == 1] = 0.9999999999999
        
    return -np.sum(y*np.log(pred), axis=1)  
    
def d_cross_entropy_loss(pred, y):
    return pred - y