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

#define BLOCKSIZE 1024
#include "CycleTimer.h"

using Eigen::MatrixXd;


//declare constant memory
__constant__ double homography[9];

__global__ void kernelWarpPerspective(int png_width, int png_height, int curr_width, int curr_height, unsigned char* out_r_device, 
                                    unsigned char* out_g_device, unsigned char* out_b_device, unsigned char* out_a_device, unsigned char* png_r, unsigned char* png_g,
                                    unsigned char* png_b, unsigned char* png_a){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= png_height || j >= png_width)
        return;

    double prod_00 = homography[0]*j + homography[3]*i + homography[6];
    double prod_10 = homography[1]*j + homography[4]*i + homography[7];
    double prod_20 = homography[2]*j + homography[5]*i + homography[8];
    
    double res_00 = (prod_00/prod_20);
    double res_10 = (prod_10/prod_20);
    if((int)res_00 >= 0 && (int)res_00 < curr_width && (int)res_10 >= 0 && res_10 < (int)curr_height){
        out_r_device[(int)res_10*curr_width+(int)res_00] = (int)png_r[i*png_width + j]; 
        out_g_device[(int)res_10*curr_width+(int)res_00] = (int)png_g[i*png_width + j]; 
        out_b_device[(int)res_10*curr_width+(int)res_00] = (int)png_b[i*png_width + j];
        out_a_device[(int)res_10*curr_width+(int)res_00] = (int)png_a[i*png_width + j]; 
    }
    
}

void dummyWarmup(){
    cudaFree(0);
}


void warpPerspective(unsigned char* png_r, unsigned char* png_g, unsigned char* png_b, unsigned char* png_a, 
    int png_width, int png_height, unsigned char* newImR, unsigned char* newImG, unsigned char* newImB, unsigned char* newImA, 
    MatrixXd H, int newIm_width, int newIm_height){

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((png_width + blockDim.x - 1) / blockDim.x, ((png_height + blockDim.y - 1) / blockDim.y));
    
    //UNAVOIDABLE TIME COST: CUDA WARMUP TIME is 0.67 seconds
    double *H_data = H.data();
   
    cudaMemcpyToSymbol(homography, H_data, sizeof(double)*9);
    
    
    unsigned char* png_r_device;
    unsigned char* png_g_device;
    unsigned char* png_b_device;
    unsigned char* png_a_device;
    cudaMalloc((void **)&png_r_device, png_height*png_width*sizeof(unsigned char));
    cudaMalloc((void **)&png_g_device, png_height*png_width*sizeof(unsigned char));
    cudaMalloc((void **)&png_b_device, png_height*png_width*sizeof(unsigned char));
    cudaMalloc((void **)&png_a_device, png_height*png_width*sizeof(unsigned char));

    cudaMemcpy(png_r_device, png_r, png_height*png_width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(png_g_device, png_g, png_height*png_width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(png_b_device, png_b, png_height*png_width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(png_a_device, png_a, png_height*png_width*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    unsigned char* out_r_device;
    unsigned char* out_g_device;
    unsigned char* out_b_device;
    unsigned char* out_a_device;
    cudaMalloc((void **)&out_r_device, newIm_width*newIm_height*sizeof(unsigned char)); //try int as well
    cudaMalloc((void **)&out_g_device, newIm_width*newIm_height*sizeof(unsigned char));
    cudaMalloc((void **)&out_b_device, newIm_width*newIm_height*sizeof(unsigned char));
    cudaMalloc((void **)&out_a_device, newIm_width*newIm_height*sizeof(unsigned char));
    // double section3E = CycleTimer::currentSeconds();
    // std::cout << "Section3 time " << section3E-section3S << std::endl;
    
    // double startTime = CycleTimer::currentSeconds();
    kernelWarpPerspective<<<gridDim, blockDim>>>(png_width, png_height, newIm_width, newIm_height,
                                                out_r_device, out_g_device, out_b_device, out_a_device, png_r_device, png_g_device,
                                                png_b_device, png_a_device);
    
    cudaMemcpy(newImR, out_r_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost); //CHECK ORDER OF ARGS HERE
    cudaMemcpy(newImG, out_g_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(newImB, out_b_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(newImA, out_a_device, newIm_width*newIm_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    cudaFree(png_r_device);
    cudaFree(png_g_device);
    cudaFree(png_b_device);
    cudaFree(png_a_device);
    cudaFree(out_r_device);
    cudaFree(out_g_device);
    cudaFree(out_b_device);
    cudaFree(out_a_device);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}

__global__ void ransacIterationDiffKernel(char* counts, char* divByZero, int threshold, double* prod, double* locs1,
                                          int prodCols, int locs1Rows){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= prodCols)
        return;
    if (prod[3*col+2] == 0){
        divByZero[col] = 1;
        return;
    }
    double diff = (double)sqrtf(powf(prod[3*col]/prod[3*col+2] - locs1[col], 2.0) + powf(prod[3*col+1]/prod[3*col+2] - locs1[locs1Rows+col], 2.0));
    if(diff < threshold){
        counts[col] = 1;
    }

}

void ransacIterationDiff(MatrixXd prod, MatrixXd locs1, int threshold, int* count){
    dim3 blockDim(BLOCKSIZE, 1);
    dim3 gridDim((prod.cols() + blockDim.x - 1) / blockDim.x, 1);

    char* countsDevice;
    cudaMalloc((void **)&countsDevice, prod.cols()*sizeof(char));
    cudaMemset(countsDevice, 0, prod.cols()*sizeof(char));

    char* divByZeroDevice;
    cudaMalloc((void **)&divByZeroDevice, prod.cols()*sizeof(char));
    cudaMemset(divByZeroDevice, 0, prod.cols()*sizeof(char));

    double* prodDevice;
    cudaMalloc((void **)&prodDevice, prod.cols()*prod.rows()*sizeof(double));
    cudaMemcpy(prodDevice, prod.data(), prod.cols()*prod.rows()*sizeof(double), cudaMemcpyHostToDevice);

    double* locs1Device;
    cudaMalloc((void **)&locs1Device, locs1.cols()*locs1.rows()*sizeof(double));
    cudaMemcpy(locs1Device, locs1.data(), locs1.cols()*locs1.rows()*sizeof(double), cudaMemcpyHostToDevice);

    ransacIterationDiffKernel<<<gridDim, blockDim>>>(countsDevice, divByZeroDevice, threshold, prodDevice, locs1Device, prod.cols(), locs1.rows());
    
    char* divByZero = (char *)calloc(prod.cols(), sizeof(char));
    char* counts = (char *)calloc(prod.cols(), sizeof(char));
    cudaMemcpy(divByZero, divByZeroDevice, prod.cols()*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(counts, countsDevice, prod.cols()*sizeof(char), cudaMemcpyDeviceToHost);

    int totalCount = 0; 
    // #pragma omp parallel for reduction(sum+: totalCount)
    for (int i = 0; i < prod.cols(); i++){
        totalCount += divByZero[i]; 
    }

    if(totalCount != 0){
        totalCount = 0; 
        // #pragma omp parallel for reduction(sum+: totalCount)
        for (int i = 0; i < prod.cols(); i++){
            totalCount += counts[i]; 
        }
    }
    *count = totalCount;
    //////////
}