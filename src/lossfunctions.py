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
    return -(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def d_bce_loss(pred, y):
    """Derivative of Binary Cross Entropy loss taken at pred.  
    
    Arguments:
        See bce_loss.

    Returns:
        See bce_loss.
    """
    return y - pred 