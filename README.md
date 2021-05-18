## Parallel Image Stitcher using ezSIFT
The parallel image stitcher developed here uses the standalone ezSIFT feature detection code to detect features in the images passed in and then creates a panorama based on the matched points. Any png image will work as an input to our program. There is no usage of OpenCV in our repo. Our code was developed on the CMU GHC machines.

### How to build
To run this code, one must have a version of CUDA >= 10.2, cmake version >= 3.20.2 and OpenMP available. 
To create the executables, follow the following instructions:
```Bash
mkdir build
cd build
cmake ..
make
```
Then you can find the built binary in the same directory. There are three versions representing our different parallelization schemes that can be run: ./image_stitching_seq, ./image_stitching_pure_openmp, and ./image_stitching_cuda, where the cuda implementation uses both OpenMP and CUDA. To run any executable, pass at least two images located in the newly created build directory in. Note that our code only works if images are fed in left-to-right order and may produce nonsensical results otherwise.

Here are two examples of how to run:
```bash
./image_stitching_seq goldengate-00.png goldengate-01.png goldengate-02.png goldengate-03.png goldengate-04.png goldengate-05.png 
./image_stitching_cuda campus_016.png campus_015.png campus_014.png
```
To run on your own unique png images, add them to the data folder and rebuild the build folder.

### License for ezSIFT
    
    Copyright 2013 Guohui Wang

[license-url]: https://github.com/robertwgh/ezSIFT/blob/master/LICENSE
[license-img]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
