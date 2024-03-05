#!/bin/sh
#BSUB -J backbone
#BSUB -o backbone%J.out
#BSUB -e backbone%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -W 1:30
#BSUB -N
# end of BSUB options

cd /zhome/2c/b/146593/Desktop/afhandling/GNN-afhandling/

# load CUDA (for GPU support)
module load cuda/11.3

# activate the virtual environment
source /zhome/2c/b/146593/Desktop/afhandling/env1/bin/activate

python src/træn.py --epoker_selvtræn 20 --epoker_efterfølgende 5 --frys_rygrad