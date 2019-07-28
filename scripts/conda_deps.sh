set -ex
conda install -c conda-forge tqdm
conda install -c conda-forge opencv
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch # add cuda90 if CUDA 9
conda install visdom dominate -c conda-forge # install visdom and dominate
