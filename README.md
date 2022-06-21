# FastFlow
Generating Images using Fast Normalizing Flows with Invertible Convolution


## Developing in ada
- create a python3.9 venv with all packages in requirments.txt installed in ~/venv/fastflow
- source env.sh
## Experiments

###Environment Setup
1. Make sure you have Anaconda or Miniconda installed.
2. Clone repo with `git clone https://github.com/aditya-v-kallappa/FastFlow/.`
3. Go into the cloned repo: `cd FastFlow`
4. Create the environment: `conda env create ff`
5. Activate the environment: `source activate ff`


### MNIST 

fastflow/fastflow_mnist.py

`set-  -n_blocks=3, block_size=32, image_size=(1, 28, 28)`

      python fastflow_mnist.py
### Imagenet 32/64

fastflow/fastflow_imagenet_multi_gpu.py

`set-  resulotion=32/64, -n_blocks=2, block_size=16, image_size=(3, 32, 32)`

      python fastflow_imagenet_multi_gpu.py      
# ToDo

