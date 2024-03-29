# ``DREAMPlaceFPGA-MP``
``DREAMPlaceFPGA-MP`` is a GPU-Accelerated Macro Placer for Large Scale Heterogeneous FPGAs using a Deep Learning Toolkit.

## <a name="team members"></a>Team Members

- Zhili Xiong, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin
- Zhixing Jiang, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin
- Rachel Selina Rajarathnam, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin
- Hanqing Zhu, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin

## <a name="supervisor"></a>Supervisor 

- David Z. Pan, Professor, [UTDA](https://www.cerc.utexas.edu/utda), ECE Department, The University of Texas at Austin

## <a name="dependencies"></a>External Dependencies

- Python 2.7 or Python 3.5/3.6/3.7

- [CMake](https://cmake.org) version 3.8.2 or later

- [Pytorch](https://pytorch.org/) 1.6/1.7/1.8
    - Other 1.0 versions may also work, but not tested.
    - **Version 2.0 or higher is not compatible!**

- [GCC](https://gcc.gnu.org/)
    - Recommend GCC 5.1 or later. 
    - Other compilers may also work, but not tested. 

- [cmdline](https://github.com/tanakh/cmdline)
    - a command line parser for C++

- [Flex](http://flex.sourceforge.net)
    - lexical analyzer employed in the bookshelf parser

- [Bison](https://www.gnu.org/software/bison)
    - parser generator employed in the bookshelf parser

- [Boost](https://www.boost.org)
    - Need to install and visible for linking

- [Limbo](https://github.com/limbo018/Limbo)
    - Integrated as a submodule: the bookshelf parser is modified for FPGAs.

- [Flute](https://doi.org/10.1109/TCAD.2007.907068)
    - Integrated as a submodule

- [CUB](https://github.com/NVlabs/cub)
    - Integrated as a git submodule

- [munkres-cpp](https://github.com/saebyn/munkres-cpp)
    - Integrated as a git submodule

- [CUDA 9.1 or later](https://developer.nvidia.com/cuda-toolkit) (Optional)
    - If installed and found, GPU acceleration will be enabled. 
    - Otherwise, only CPU implementation is enabled. 

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 6.0, 6.1, 7.0, 7.5, and 8.0. 
    - Please check the [compatibility](https://developer.nvidia.com/cuda-gpus) of the GPU devices. 
    - The default compilation target is compatibility 6.0. This is the minimum requirement and lower compatibility is not supported for the GPU feature. 

- [Cairo](https://github.com/freedesktop/cairo) (Optional)
    - If installed and found, the plotting functions will be faster by using C/C++ implementation. 
    - Otherwise, python implementation is used.
 
## Build with Docker

You can use the Docker container to avoid building all the dependencies yourself.

1.  Install Docker on [Linux](https://docs.docker.com/install/).(Win and Mac are not tested)
2.  To enable the GPU features, install [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker); otherwise, skip this step.
3.  Download the DREAMPlaceFPGA-MP-main.zip, and navigate to the repository
    ```
    unzip DREAMPlaceFPGA-MP-main.zip
    cd DREAMPlaceFPGA-MP
    ```
4. Get the docker image using one of the options
    - Option 1: pull the image from the cloud
    ```
    docker pull zhixingjiang/dreamplace_fpga:v1.0
    ```
    - Option 2: build the image locally
    ```
    docker build . --file Dockerfile --tag <username>/dreamplace_fpga:1.0
    ```
    replace `<username>` with a username, for instance 'utda_macro_placer'.
5. Enter bash environment of the container.
    Mount the repo and all the Designs into the Docker, which allows the Docker container to directly access and modify these files

    To run on a Linux machine without GPU:
    ```
    docker run -it -v $(pwd):/DREAMPlaceFPGA-MP -v <path_to_designs_directory>:/Designs <username>/dreamplace_fpga:1.0 bash
    ```
    To run on a Linux machine with GPU: (Docker verified on NVIDIA GPUs with compute capability 6.1, 7.5, and 8.0)
    ```
    docker run --gpus 1 -it -v $(pwd):/DREAMPlaceFPGA-MP -v <path_to_designs_directory>:/Designs <username>/dreamplace_fpga:1.0 bash
    ```
    Provide complete path to the designs directory for <path_to_designs_directory>, which contains `Design_1`, `Design_2`, `Design_3`, etc...

    For example to run on a Linux machine without GPU:
    ```
    docker run -it -v $(pwd):/DREAMPlaceFPGA-MP -v $(pwd)/../Designs:/Designs utda_macro_placer/dreamplace_fpga:1.0 bash
    ```
6. Go to the `DREAMPlaceFPGA-MP` directory in the Docker
    ```
    cd /DREAMPlaceFPGA-MP
    ```
7. Create a build directory and install the package
    ```
    rm -rf build
    mkdir build 
    cd build 
    cmake .. -DCMAKE_INSTALL_PREFIX=/DREAMPlaceFPGA-MP -DPYTHON_EXECUTABLE=$(which python)
    make
    make install
    ```
    Note: When there are changes to packages or parser code, it is necessary to delete contents of ***build*** directory for a clean build and proper operation.
    If there is no change to code, there is no need to delete the build directory using `rm -r build`.
    ```
    rm -r build
    ```
8.  To run the UTDA_macro_placer on a design
    Go to the ***design directory*** containing the bookshelf input files, and run:
    ```
    source <path_to_root_dir>/run_mlcad_design.sh <path_to_root_dir> <gpu_flag>
    ```
    In our case the path to root dir is just `/DREAMPlaceFPGA-MP`, replace gpu_flag with 1 or 0, 1 is using GPU, 0 is using CPU
    ```
    source /DREAMPlaceFPGA-MP/run_mlcad_design.sh /DREAMPlaceFPGA-MP <gpu_flag>
    ```
    

## Build without Docker

1.  To pull git submodules in the root directory
    ```
    git submodule init
    git submodule update
    ```
    
    Or alternatively, pull all the submodules when cloning the repository. 
    ```
    git clone --recursive https://github.com/zhilix/DREAMPlaceFPGA-MP.git
    ```

2.  To install Python dependency 
    At the root directory:
    ```
    pip install -r requirements.txt 
    ```
    > For example, if the repository was cloned in directory ***~/Downloads***, then the root directory is ***~/Downloads/DREAMPlaceFPGA-MP***
    
    > You can also use a [python virtual environment](https://docs.python.org/3/library/venv.html) to install all the required packages to run ``DREAMPlaceFPGA-MP``

3.  To Build 
    At the root directory, 
    ```
    mkdir build 
    cd build 
    cmake .. -DCMAKE_INSTALL_PREFIX=<path_to_root_dir>
    make
    make install
    ```
    Third party submodules are automatically built except for [Boost](https://www.boost.org).
    
    > For example,
    
    > ***~/Downloads/DREAMPlaceFPGA-MP:*** *mkdir build; cd build*
    
    > ***~/Downloads/DREAMPlaceFPGA-MP/build:***  *cmake . . -DCMAKE_INSTALL_PREFIX=~/Downloads/DREAMPlaceFPGA-MP*
    
    > ***~/Downloads/DREAMPlaceFPGA-MP/build:*** *make; make install*
    
    > The directory ***~/Downloads/DREAMPlaceFPGA-MP/build*** is the install dir
    
    When there are changes to packages or parser code, it is necessary to delete contents of ***build*** directory for a clean build and proper operation.
    ```
    rm -r build
    ```
    > For example,
    
    > ***~/Downloads/DREAMPlaceFPGA-MP:*** *rm -r build*

4.  Running UTDA_macro_placer
    Before running, ensure that all python dependent packages have been installed. 
    Go to the ***benchmark directory***, and run:
    ```
    source <path_to_root_dir>/run_mlcad_design.sh <path_to_root_dir> <gpu_flag>
    ```
    > Run from ***~/Downloads/DREAMPlaceFPGA-MP/benchmarks/mlcad2023_benchmarks/Design_181*** directory
    
    For example, to run on GPU: 
    > ***~/Downloads/Designs/Design_181:*** *source ~/Downloads/DREAMPlaceFPGA-MP/run_mlcad_design.sh ~/Downloads/DREAMPlaceFPGA-MP 1*

### Optional Cmake Options
    Here are the available options for CMake. 
    - CMAKE_INSTALL_PREFIX: installation or root directory
        - Example ```cmake -DCMAKE_INSTALL_PREFIX=path/to/root/directory```
    - CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
        - Example ```cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60```
    - CMAKE_CXX_ABI: 0|1 for the value of _GLIBCXX_USE_CXX11_ABI for C++ compiler, default is 0. 
        - Example ```cmake -DCMAKE_CXX_ABI=0```
        - It must be consistent with the _GLIBCXX_USE_CXX11_ABI for compling all the C++ dependencies, such as Boost and PyTorch. 
        - PyTorch in default is compiled with _GLIBCXX_USE_CXX11_ABI=0, but in a customized PyTorch environment, it might be compiled with _GLIBCXX_USE_CXX11_ABI=1. 

## <a name="copyright"></a>Copyright

This software is released under *BSD 3-Clause "New" or "Revised" License*. Please refer to [LICENSE](./LICENSE) for details.
