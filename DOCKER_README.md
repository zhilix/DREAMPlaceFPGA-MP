# DREAMPlace_MLCAD
- [How to Build](#how-to-build)

## Build with Docker

You can use the Docker container to avoid building all the dependencies yourself.

1.  Install Docker on [Linux](https://docs.docker.com/install/).(Win and Mac are not tested)
2.  git clone the git repo, and navigate to the repository
    ```
    git clone --recursive https://github.com/zhilix/DREAMPlaceFPGA_mlcad.git
    cd DREAMPlaceFPGA_mlcad
    ```
3. build the docker image
    ```
    docker build . --file Dockerfile --tag your_username/dreamplace_fpga:1.0
    ```
    replace `your_username` with your name.
4. Enter bash environment of the container.
    Mount the repo and all the Designs into the Docker, which allows the Docker container to directly access and modify these files
    ```
    docker run -it   -v $(pwd):/DREAMPlaceFPGA_mlcad   -v $(pwd)/../Designs:/Designs   your_username/dreamplace_fpga:1.0 bash
    ```
    Replace `your_username` with your name
    Replace `$(pwd)/../Designs` with the absolute path of the Designs folder, which containing `Design_1`, `Design_2`, `Design 3`, etc...
5. go to the `DREAMPlaceFPGA_mlcad` directory
    ```
    cd /DREAMPlaceFPGA_mlcad
    ```
6. create a build directory and install the package
    ```
    rm -rf build
    mkdir build 
    cd build 
    cmake .. -DCMAKE_INSTALL_PREFIX=/DREAMPlaceFPGA_mlcad -DPYTHON_EXECUTABLE=$(which python)
    make
    make install
    ```
    Note: When there are changes to packages or parser code, it is necessary to delete contents of ***build*** directory for a clean build and proper operation.
    ```
    rm -r build
    ```


