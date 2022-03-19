from util import *
import numpy as np


# NOTE: important to mention in the paper also, that if kernel[-1,-1] is not 1, it can either result in 
# too small values in the output or large numerical errors. 
# TODO: Need to arrive at a theoretical justification for above.


# if kernel is not normalized, the values in the output becomes very large 
# resulting in large numerical errors
kernel = (np.random.random((5, 5)) - 0.5)/25.0
kernel[-1, -1] = 1.0 # NOTE: its essential that this value is not too small. Othewise it results in large numerical errors
test_inverse(
    input=np.random.random((2,512, 512)) - 0.5,
    kernel=kernel)
    
    
# there some error with larger batchsize or image size. possibly some issues with blocks