from util import *

test_inverse(
    input=[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
    kernel=[
            [0,0],
            [0,1]
    ])
    

test_inverse(
    input=[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
    kernel=[
            [1,0],
            [0,1]
    ])


test_inverse(
    input=[
            [1,2,3],
            [5,6,7],
            [9,10,11]
        ],
    kernel=[
            [1,0],
            [0,1]
    ])


test_inverse(
    input=[
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
            [13,14,15,16]
        ],
    kernel=[
            [1,0,0],
            [0,1,0],
            [0,0,1],
    ])
