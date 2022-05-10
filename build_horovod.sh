#!/bin/bash
export HOROVOD_CUDA_HOME=/usr/local/cuda-10.2/
export HOROVOD_WITHOUT_MPI=1
export HOROVOD_WITH_GLOO=1
export HOROVOD_CPU_OPERATION=GLOO
export HOROVOD_GPU=CUDA
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITHOUT_NVTX=1

PYTHON_PATH=$(which python3)
${PYTHON_PATH} setup.py build_ext 2>&1 | tee compile.log; [[ ${PIPESTATUS[0]} == 0 ]] || exit

${PYTHON_PATH} setup.py bdist_wheel -d build_pip || exit
# Return 0 status if all finished
exit 0
