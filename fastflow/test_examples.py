import torch

from test_layers import *

# test_PaddedConv2d(
#     x = torch.tensor(
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ], dtype=torch.float32
#     ).reshape(1, 1, 3, 3), 

#     w = torch.tensor(
#         [
#             [0, 1],
#             [1, 0]
#         ], dtype=torch.float32
#     ).reshape(1, 1, 2, 2),

#     order='BL',

#     print_answer=False
# )



# test_PaddedConv2d(
#     x = torch.tensor(
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ], dtype=torch.float32
#     ).reshape(1, 1, 3, 3), 

#     # w = torch.tensor(
#     #     [
#     #         [1, 0],
#     #         [0, 2]
#     #     ], dtype=torch.float32
#     # ).reshape(1, 1, 2, 2),
#     w=None,
#     order='BR',

#     print_answer=False
# )


# test_PaddedConv2d(
#     x = torch.tensor(
#     [
#         [ 1.,  2.,  3.],
#         [ 8., 11.,  6.],
#         [17., 20.,  9.]
#     ], dtype=torch.float32
#     ).reshape(1, 1, 3, 3), 

#     w = torch.tensor(
#         [
#             [0, 2],
#             [1, 0]
#         ], dtype=torch.float32
#     ).reshape(1, 1, 2, 2),

#     order='TR',
#     is_input=False,
#     print_answer=False
# )

# test_FastFlowUnit(
#     x = torch.rand((1, 16, 7, 7)),
#     kernel_size=(2, 2),
#     block_size=20,
#     is_input=True,
#     print_answer=False
# )

# test_FastFlowUnit(
#     x = torch.rand((10, 48, 10, 10)),
#     kernel_size=(3, 3),
#     is_input=True,
#     print_answer=False
# )

# x = torch.randn(28 * 28).reshape(-1, 28, 28)
# test_Dequantization(
#     x=x,
#     size=x.shape,
#     is_input=False
# )

# test_Normalization(
#     x = torch.rand((10, 48, 10, 10)),
#     translation=10,
#     scale=0.1,
#     is_input=False
# )


# alpha = 1e-6
# test_Normalization(
#     x = torch.rand((10, 48, 10, 10)),
#     translation=-alpha,
#     scale=1 / (1 - 2 * alpha),
#     is_input=False
# )

# test_LogitTransform(
#     x = torch.rand((10, 48, 100, 100))
# )

# test_LogitTransform(
#     x = torch.rand((10, 16, 7, 7)),
#     is_input=False
# )

# test_Squeeze(
#     x = torch.rand((100, 16, 20, 10)),       #H, W have to be even
#     is_input=False
# )

