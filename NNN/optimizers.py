import numpy as np
        
def vanilla(prev_update, grad, mom_frac, lr):
    """No Optimization applied.
    
    Arguments:
        prev_update {np.array} -- Vec of previous updates made 
        grad {np.array} -- Current gradient.
        mom_frac {[type]} -- How much of the previous update to keep as momentum.
        lr {num} -- Learning rate.
    
    Returns:
        np.array -- Update to the parameters. 
    """

    return grad * lr

def momentum(prev_update, grad, mom_frac, lr):
    """
    Add a portion(mom_frac) of the previous timesteps update vector to the currently calculated gradient.
    """
    return prev_update * mom_frac + lr * grad

def nesterov_momentum():
    pass





