#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3072
#SBACTH --job-name=mnist_glow_job
#SBATCH --output=mnist_glow.%j.out

cd ~
python check_cuda.py
source ~/anaconda3/etc/profile.d/conda.sh
source activate py39

cd ~/Research/NormalizingFlows/FastFlow/fastflow
source ../env.sh

echo "Training the model"
python glow_mnist.py
