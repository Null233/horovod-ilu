# Horovod-corex
## Intro
Horovod-corex is based on native horovod source code, but use Iluvatar compiler(ilcc) to compile to adapt Iluvatar SDKv1.1. All changes are related to compilation and installation on BI environment.

## How to Build and Install
To build on BI environment, please follow the listed steps below:
1. Check your cmake version `cmake --version`, all changes and compilations are based on `cmake 3.16`, upgrade cmake if needed.
2. Get the source code from horovod official website: `git clone https://github.com/horovod/horovod.git`
3. Get the submodules: `git submodule update --recursive --init`(Proxy may be needed)
4. Change the CWD to the repo: `cd horovod`
5. Reset to commit id: `git reset --hard 93a2f2583ed63391a904aaeb03b602729be90f15` (Or download horovod v0.22.0)
6. Apply the patch: `git am Horovod-corex.patch`
7. Set your cuda path and corex SDK installation path in `build_horovod.sh` by `vim build_horovod.sh` if using vim as your text editor. In `build_horovod.sh` script, `$TARGET_DIR` denotes your corex SDK installation path, `$TARGET_CUDA_DIR` denotes your cuda installation path. Other environment variables can be added to active more horovod functionalities according to horovod official installation guide.
8. Run command `bash build_horovod.sh` to auto build.
9. Run command `bash install_horovod.sh` to auto install.
10. `export PATH=${HOME}/.local/bin:$PATH` to enable `horovodrun` command, or you can use `mpirun` instead of `horovodrun`.

## How to Clean
To clean, run command `bash clean_horovod.sh`

## Tests
Official test samples are under `horovod/examples`. All rules of use keep the same as official horovod.
Example:
`cd examples/pytorch`
`horovodrun -np 2 python3 pytorch_mnist.py`
