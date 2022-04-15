# FastFlow
Generating Images using Fast Normalizing Flows with Invertible Convolution


## Developing in ada
- create a python3.9 venv with all packages in requirments.txt installed in ~/venv/fastflow
- source env.sh
## Experiments
### Imagenet 32/64

fastflow/fastflow_imagenet_multi_gpu.py

set-  resulotion=32/64, -n_blocks=2, block_size=16, image_size=(3, 32, 32)

      python fastflow_imagenet_multi_gpu.py
# ToDo

