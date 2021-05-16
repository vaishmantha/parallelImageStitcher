#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <list>
// #include <eigen/Eigen/Core>
#include <eigen/Eigen/Dense>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

// #define BLOCKSIZE 1024
#include "CycleTimer.h"

using Eigen::MatrixXd;


//declare constant memory
__constant__ double homography[9];

__global__ void kernelWarpPerspective(int png_width, int png_height, int curr_width, int curr_height, unsigned char* out_r_device, 
                                    unsigned char* out_g_device, unsigned char* out_b_device, unsigned char* out_a_device, unsigned char* png_r, unsigned char* png_g,
                                    unsigned char* png_b, unsigned char* png_a){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i > png_height || j > png_width)
        return;

    double prod_00 = homography[0]*j + homography[3]*i + homography[6];
    double prod_10 = homography[1]*j + homography[4]*i + homography[7];
    double prod_20 = homography[2]*j + homography[5]*i + homography[8];
    
    int res_00 = (int)(prod_00/prod_20);
    int res_10 = (int)(prod_10/prod_20);
    if(res_00 >= 0 && res_00 < curr_width && res_10 >= 0 && res_10 < curr_height){
        out_r_device[res_10*curr_width+res_00] = png_r[i*png_width + j]; 
        out_g_device[res_10*curr_width+res_00] = png_g[i*png_width + j]; 
        out_b_device[res_10*curr_width+res_00] = png_b[i*png_width + j];
        out_a_device[res_10*curr_width+res_00] = png_a[i*png_width + j]; 
    }
    
}

void warpPerspective(unsigned char* png_r, unsigned char* png_g, unsigned char* png_b, unsigned char* png_a, 
    int png_width, int png_height, unsigned char* newImR, unsigned char* newImG, unsigned char* newImB, unsigned char* newImA, 
    MatrixXd H, int newIm_width, int newIm_height){
    double overallStartTime = CycleTimer::currentSeconds();
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((png_width + blockDim.x - 1) / blockDim.x, ((png_height + blockDim.y - 1) / blockDim.y));
    
    double section1S = CycleTimer::currentSeconds();
    // double* H_device;
    double *H_data = H.data();
    cudaMemcpyToSymbol(homography, H_data, sizeof(double)*9);
    double section1E = CycleTimer::currentSeconds();
    std::cout << "Section1 time " << section1E-section1S << std::endl;
    
    double section2S = CycleTimer::currentSeconds();
    unsigned char* png_r_device;
    unsigned char* png_g_device;
    unsigned char* png_b_device;
    unsigned char* png_a_device;
    cudaMalloc((void **)&png_r_device, png_height*png_width*sizeof(char));
    cudaMalloc((void **)&png_g_device, png_height*png_width*sizeof(char));
    cudaMalloc((void **)&png_b_device, png_height*png_width*sizeof(char));
    cudaMalloc((void **)&png_a_device, png_height*png_width*sizeof(char));

    cudaMemcpy(png_r_device, png_r, png_height*png_width*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(png_g_device, png_g, png_height*png_width*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(png_b_device, png_b, png_height*png_width*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(png_a_device, png_a, png_height*png_width*sizeof(char), cudaMemcpyHostToDevice);
    double section2E = CycleTimer::currentSeconds();
    std::cout << "Section2 time " << section2E-section2S << std::endl;
    
    double section3S = CycleTimer::currentSeconds();
    unsigned char* out_r_device;
    unsigned char* out_g_device;
    unsigned char* out_b_device;
    unsigned char* out_a_device;
    cudaMalloc((void **)&out_r_device, newIm_width*newIm_height*sizeof(unsigned char)); //try int as well
    cudaMalloc((void **)&out_g_device, newIm_width*newIm_height*sizeof(unsigned char));
    cudaMalloc((void **)&out_b_device, newIm_width*newIm_height*sizeof(unsigned char));
    cudaMalloc((void **)&out_a_device, newIm_width*newIm_height*sizeof(unsigned char));
    double section3E = CycleTimer::currentSeconds();
    std::cout << "Section3 time " << section3E-section3S << std::endl;
    
    double startTime = CycleTimer::currentSeconds();
    kernelWarpPerspective<<<gridDim, blockDim>>>(png_width, png_height, newIm_width, newIm_height,
                                                out_r_device, out_g_device, out_b_device, out_a_device, png_r_device, png_g_device,
                                                png_b_device, png_a_device);
    double endTime = CycleTimer::currentSeconds();
    std::cout << "Actual kernel time " << endTime-startTime << std::endl;
    

    double lMemcpyTime = CycleTimer::currentSeconds();
    cudaMemcpy(newImR, out_r_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost); //CHECK ORDER OF ARGS HERE
    cudaMemcpy(newImG, out_g_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(newImB, out_b_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(newImA, out_a_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    double lEndMemcpyTime = CycleTimer::currentSeconds();
    std::cout << "Memcpy kernel time " << lEndMemcpyTime-lMemcpyTime << std::endl;
    
    double section4S = CycleTimer::currentSeconds();
    cudaFree(png_r_device);
    cudaFree(png_g_device);
    cudaFree(png_b_device);
    cudaFree(png_a_device);
    cudaFree(out_r_device);
    cudaFree(out_g_device);
    cudaFree(out_b_device);
    cudaFree(out_a_device);
    double section4E = CycleTimer::currentSeconds();
    std::cout << "Section4 time " << section4E-section4S << std::endl;
    
    double overallEndTime = CycleTimer::currentSeconds();
    std::cout << "Overall warp persp time " << overallEndTime-overallStartTime << std::endl;

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}


