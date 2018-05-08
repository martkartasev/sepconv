#!/bin/bash

#
# Created by: Simon Niklaus, Long Mai, Feng Liu
# https://github.com/sniklaus/pytorch-sepconv
#

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

nvcc -c -o src/SeparableConvolution_kernel.o src/SeparableConvolution_kernel.cu --gpu-architecture=compute_52 --gpu-code=compute_52 --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC

python install.py
