# ``UTDA_macro_placer``
``UTDA_macro_placer`` is a GPU-Accelerated Macro Placer for Large Scale Heterogeneous FPGAs using a Deep Learning Toolkit.

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
    - Code has been tested on GPUs with compute compatibility 6.0, 7.0, and 7.5. 
    - Please check the [compatibility](https://developer.nvidia.com/cuda-gpus) of the GPU devices. 
    - The default compilation target is compatibility 6.0. This is the minimum requirement and lower compatibility is not supported for the GPU feature. 

- [Cairo](https://github.com/freedesktop/cairo) (Optional)
    - If installed and found, the plotting functions will be faster by using C/C++ implementation. 
    - Otherwise, python implementation is used.
 
## Build with Docker

You can use the Docker container to avoid building all the dependencies yourself.

1.  Install Docker on [Linux](https://docs.docker.com/install/).(Win and Mac are not tested)
2.  To enable the GPU features, install [NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker); otherwise, skip this step.
3.  git clone the git repo, and navigate to the repository
    ```
    git clone --recursive https://github.com/zhilix/DREAMPlaceFPGA_mlcad.git
    cd DREAMPlaceFPGA_mlcad
    ```
4. build the docker image
    - Option 1: pull the image from the cloud
    ```
    docker pull zhixingjiang/dreamplace_fpga:v1.0
    ```
    - Option 2: build the image locally
    ```
    docker build . --file Dockerfile --tag your_username/dreamplace_fpga:1.0
    ```
    
    replace `your_username` with your name.
6. Enter bash environment of the container.
    Mount the repo and all the Designs into the Docker, which allows the Docker container to directly access and modify these files

    Run without GPU on Linux.
    ```
    docker run -it -v $(pwd):/DREAMPlaceFPGA_mlcad -v /Your_Designs_Directory:/Designs your_username/dreamplace_fpga:1.0 bash
    ```
    Run with GPU on Linux. (not tested yet)
    ```
    docker run --gpus 1 -it -v $(pwd):/DREAMPlaceFPGA_mlcad -v /Your_Designs_Directory:/Designs your_username/dreamplace_fpga:1.0 bash
    ```
    Replace `your_username` with your name if using option 2.
    Replace `your_username` with your `zhixingjiang` if using option 1. 
    Replace `/Your_Designs_Directory` with the path of the Designs folder, which containing `Design_1`, `Design_2`, `Design_3`, etc... For example:
    ```
    docker run -it -v $(pwd):/DREAMPlaceFPGA_mlcad -v $(pwd)/../Designs:/Designs your_username/dreamplace_fpga:1.0 bash
    ```
8. go to the `DREAMPlaceFPGA_mlcad` directory
    ```
    cd /DREAMPlaceFPGA_mlcad
    ```
9. create a build directory and install the package
    ```
    rm -rf build
    mkdir build 
    cd build 
    cmake .. -DCMAKE_INSTALL_PREFIX=/DREAMPlaceFPGA_mlcad -DPYTHON_EXECUTABLE=$(which python)
    make
    make install
    ```
    Note: When there are changes to packages or parser code, it is necessary to delete contents of ***build*** directory for a clean build and proper operation. if no change to the code, no need to run rm -r build.
    ```
    rm -r build
    ```
10.  Running UTDA_macro_placer
    Before running, ensure that all python dependent packages have been installed. 
    Go to the ***benchmark directory***, and run:
    ```
    source <path_to_root_dir>/run_mlcad_design.sh <path_to_root_dir> <gpu_flag>
    ```
    In our case the path to root dir is just `/DREAMPlaceFPGA_mlcad`, replace gpu_flag with 1 or 0, 1 is using GPU, 0 is using CPU
    ```
    source /DREAMPlaceFPGA_mlcad/run_mlcad_design.sh /DREAMPlaceFPGA_mlcad <gpu_flag>
    ```
    

## Build without Docker

1.  To pull git submodules in the root directory
    ```
    git submodule init
    git submodule update
    ```
    
    Or alternatively, pull all the submodules when cloning the repository. 
    ```
    git clone --recursive https://github.com/zhilix/DREAMPlaceFPGA_mlcad.git
    ```

2.  To install Python dependency 
    At the root directory:
    ```
    pip install -r requirements.txt 
    ```
    > For example, if the repository was cloned in directory ***~/Downloads***, then the root directory is ***~/Downloads/DREAMPlaceFPGA_mlcad***
    
    > You can also use a [python virtual environment](https://docs.python.org/3/library/venv.html) to install all the required packages to run ``DREAMPlaceFPGA_mlcad``

3.  To Build 
    At the root directory, 
    ```
    mkdir build 
    cd build 
    cmake .. -DCMAKE_INSTALL_PREFIX=path_to_root_dir
    make
    make install
    ```
    Third party submodules are automatically built except for [Boost](https://www.boost.org).
    
    > For example,
    
    > ***~/Downloads/DREAMPlaceFPGA_mlcad:*** *mkdir build; cd build*
    
    > ***~/Downloads/DREAMPlaceFPGA_mlcad/build:***  *cmake . . -DCMAKE_INSTALL_PREFIX=~/Downloads/DREAMPlaceFPGA_mlcad*
    
    > ***~/Downloads/DREAMPlaceFPGA_mlcad/build:*** *make; make install*
    
    > The directory ***~/Downloads/DREAMPlaceFPGA_mlcad/build*** is the install dir
    
    When there are changes to packages or parser code, it is necessary to delete contents of ***build*** directory for a clean build and proper operation.
    ```
    rm -r build
    ```
    > For example,
    
    > ***~/Downloads/DREAMPlaceFPGA_mlcad:*** *rm -r build*

4.  Cmake Options 
    Here are the available options for CMake. 
    - CMAKE_INSTALL_PREFIX: installation or root directory
        - Example ```cmake -DCMAKE_INSTALL_PREFIX=path/to/root/directory```
    - CMAKE_CUDA_FLAGS: custom string for NVCC (default -gencode=arch=compute_60,code=sm_60)
        - Example ```cmake -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_60,code=sm_60```
    - CMAKE_CXX_ABI: 0|1 for the value of _GLIBCXX_USE_CXX11_ABI for C++ compiler, default is 0. 
        - Example ```cmake -DCMAKE_CXX_ABI=0```
        - It must be consistent with the _GLIBCXX_USE_CXX11_ABI for compling all the C++ dependencies, such as Boost and PyTorch. 
        - PyTorch in default is compiled with _GLIBCXX_USE_CXX11_ABI=0, but in a customized PyTorch environment, it might be compiled with _GLIBCXX_USE_CXX11_ABI=1. 


5.  Running UTDA_macro_placer
    Before running, ensure that all python dependent packages have been installed. 
    Go to the ***benchmark directory***, and run:
    ```
    source <path_to_root_dir>/run_mlcad_design.sh <path_to_root_dir> <gpu_flag>
    ```
    > Run from ***~/Downloads/DREAMPlaceFPGA_mlcad/benchmarks/mlcad2023_benchmarks/Design_181*** directory
    
    For example, to run on GPU: 
    > ***~/Downloads/Designs/Design_181:*** *source ~/Downloads/DREAMPlaceFPGA_mlcad/run_mlcad_design.sh ~/Downloads/DREAMPlaceFPGA_mlcad 1*


## <a name="copyright"></a>Copyright

This software is released under *BSD 3-Clause "New" or "Revised" License*. Please refer to [LICENSE](./LICENSE) for details.
