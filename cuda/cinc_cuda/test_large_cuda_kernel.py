from util import *
import numpy as np


# if kernel is not normalized, the values in the output becomes very large 
# resulting in large numerical errors
kernel = (np.random.random((10, 10)) - 0.5)/100.0
kernel[-1, -1] = 1.0
test_inverse(
    input=np.random.random((1024, 1024)) - 0.5,
    kernel=kernel)
    