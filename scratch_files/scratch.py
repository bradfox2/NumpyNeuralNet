import numpy as np
from NNN.activation_functions import softmax

a = np.random.random([10,10])
x = softmax(a)
y = np.eye(10,dtype=np.int8)[np.random.choice(10, 10)]

np.sum(y*np.log(x), axis=1)