## Parallel Image Stitcher

The goal of our project was to build an image stitcher that takes input a sequence of rotation variant images and output a panoramic image. The image stitching pipeline can be broken down into the following stages: keypoint detection, keypoint matching, homography calculation via RANSAC, image warping, image placement onto final panorama. We used the standalone ezSIFT feature detection code for the first two phases of the pipeline. 

There is no usage of OpenCV in our repo. Our code was developed on the CMU GHC machines.

### Summary
We implemented an image stitcher for png images using OpenMP and enhanced sections of the pipeline with CUDA. Our initial goal was to have this work on videos, but due to issues with OpenCV, we had to manually process images on our own and therefore were not able to process videos. However, our algorithm is easily amenable to working with videos. We achieved a speedup of about 5x on 8 cores over our sequential implementation on average for any number and any size of input images fed into the program.


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
./image_stitching_cuda campus_01.png campus_02.png campus_03.png
```
To run on your own unique png images, add them to the data folder and rebuild the build folder.

### License for ezSIFT
    
    Copyright 2013 Guohui Wang

[license-url]: https://github.com/robertwgh/ezSIFT/blob/master/LICENSE
[license-img]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
