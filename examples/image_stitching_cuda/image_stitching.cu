#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <list>
#include <eigen/Eigen/Dense>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

using Eigen::MatrixXd;

double cudaFindPeaks() {
    int *device_input;
    cudaMalloc((void **)&device_input, 2 * sizeof(int));
    // int *device_input;
    // int *device_output;
    // int rounded_length = nextPow2(length);
    // cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    // cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    // cudaMemcpy(device_input, input, length * sizeof(int), 
    //            cudaMemcpyHostToDevice);

    // double startTime = CycleTimer::currentSeconds();
    
    // int result = find_peaks(device_input, length, device_output);

    // cudaDeviceSynchronize();
    // double endTime = CycleTimer::currentSeconds();

    // *output_length = result;

    // cudaMemcpy(output, device_output, length * sizeof(int),
    //            cudaMemcpyDeviceToHost);

    // cudaFree(device_input);
    // cudaFree(device_output);
    std::cout << "new Running this" << std::endl;

    return 0;
}

// __global__ void kernelInterpolate(MatrixXd* resImg, MatrixXd* copyRes, int start_i, int start_j, double max_x, double max_y){
//     int threadIdxX = blockIdx.x * blockDim.x + threadIdx.x; 
//     int threadIdxY = blockIdx.y * blockDim.y + threadIdx.y; 

//     int i = threadIdxX + start_i;
//     int j = threadIdxY + start_j; 
//     if(i < (int)max_y && j < (int)max_x) {
//         if((*resImg)(i, j) == 0){
//             if (i+1 < max_y && (*copyRes)(i+1,j) != 0){ // && i-1 >=fmax(min_y,0) && j+1 < max_x && j-1 >=fmax(min_x,0) ){
//                 (*resImg)(i, j) = (*copyRes)(i+1,j);
//             }else if(i-1 >= start_i && (*copyRes)(i-1,j) != 0){
//                 (*resImg)(i, j) = (*copyRes)(i-1,j);
//             }else if(j+1 < max_x && (*copyRes)(i,j+1) != 0){
//                 (*resImg)(i, j) = (*copyRes)(i,j+1);
//             }else if(j-1 >= start_j && (*copyRes)(i,j-1) != 0){
//                 (*resImg)(i,j) = (*copyRes)(i,j-1);
//             }else if(i+1 < max_y && j+1 < max_x && (*copyRes)(i+1,j+1)){
//                 (*resImg)(i,j) = (*copyRes)(i+1,j+1);
//             }else if(i-1 >= start_i && j+1 < max_x && (*copyRes)(i-1,j+1)){
//                 (*resImg)(i,j) = (*copyRes)(i-1,j+1);
//             }else if(i+1 < max_y && j-1 >= start_j && (*copyRes)(i+1,j-1)){
//                 (*resImg)(i,j) = (*copyRes)(i+1,j-1);
//             }else if(i-1 >= start_i && j-1 >=start_j && (*copyRes)(i-1,j-1)){
//                 (*resImg)(i,j) = (*copyRes)(i-1,j-1);
//             }
//         }
//     }
// }

__global__ void kernelCompose(MatrixXd *resImg, MatrixXd *newImage, int start_i, int start_j, double max_x, double max_y){
    int threadIdxX = blockIdx.x * blockDim.x + threadIdx.x; 
    int threadIdxY = blockIdx.y * blockDim.y + threadIdx.y; 

    int i = threadIdxX + start_i;
    int j = threadIdxY + start_j; 
    if(i < (int)max_y && j < (int)max_x) {
        if ((*resImg)(i,j) == 0){
            (*resImg)(i,j) = (*newImage)(i,j);
        }
        if ((*resImg)(i,j) != 0 && (*newImage)(i, j) != 0){
            (*resImg)(i,j) = fmax((*newImage)(i,j), (*resImg)(i,j));
        }
    }
}

void placeImage(MatrixXd newImage, MatrixXd* resImg, double min_x, double min_y, double max_x, double max_y){
    // int w = newImage.cols();
    // int h = newImage.rows();
    // printf("w: %d, h: %d", w, h);
    int start_i = (int)fmax(min_y,0);
    int start_j = (int)fmax(min_x,0);
    MatrixXd *resImg_device; 
    MatrixXd *newImage_device; 

    std::cout << "Size of matrixxd " << sizeof(MatrixXd) << std::endl;
    cudaMalloc((void **)&resImg_device, sizeof(double)*sizeof(resImg.rows()*resImg.cols()));
    cudaMalloc((void **)&newImage_device, sizeof(double)*sizeof(newImage.rows()*newImage.cols()));
    cudaMemcpy(resImg_device, resImg, sizeof(double)*sizeof(resImg.rows()*resImg.cols()), cudaMemcpyHostToDevice);
    cudaMemcpy(newImage_device, &newImage, sizeof(double)*sizeof(newImage.rows()*newImage.cols()), cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((((int)max_y) - start_i + blockDim.x - 1) / blockDim.x, (((int)max_x) - start_j + blockDim.y - 1) / blockDim.y);
    kernelCompose<<<gridDim, blockDim>>>(resImg_device, newImage_device, start_i, start_j, max_x, max_y); 
    cudaMemcpy(resImg, resImg_device, sizeof(double)*sizeof(resImg.rows()*resImg.cols()), cudaMemcpyDeviceToHost);
    // std::cout << *resImg << std::endl;
    // #pragma omp parallel for schedule(dynamic)
    // for (int i = start_i; i < (int)max_y; i++){ //access as row col
    //     for (int j = start_j; j < (int)max_x; j++){
    //         if ((*resImg)(i,j) == 0){
    //             (*resImg)(i,j) = newImage(i,j);
    //         }
    //         if ((*resImg)(i,j) != 0 && newImage(i, j) != 0){
    //             (*resImg)(i,j) = fmax(newImage(i,j), (*resImg)(i,j));
    //         }
    //     }
    // }
    
    
    MatrixXd copyRes = (*resImg);
    // dim3 blockDim(16, 16, 1);
    // dim3 gridDim((((int)max_y) - start_i + blockDim.x - 1) / blockDim.x, (((int)max_x) - start_j + blockDim.y - 1) / blockDim.y);
    // MatrixXd *resImg_device; 
    // MatrixXd *copyRes_device; 

    // cudaMalloc((void **)&resImg_device, sizeof(MatrixXd));
    // cudaMalloc((void **)&copyRes_device, sizeof(MatrixXd));
    // cudaMemcpy(resImg_device, resImg, sizeof(MatrixXd), cudaMemcpyHostToDevice);
    // cudaMemcpy(copyRes_device, &copyRes, sizeof(MatrixXd), cudaMemcpyHostToDevice);
    
    // kernelInterpolate<<<gridDim, blockDim>>>(resImg_device, copyRes_device, start_i, start_j, max_x, max_y);

    // cudaMemcpy(resImg, resImg_device, sizeof(MatrixXd), cudaMemcpyDeviceToHost);

    #pragma omp parallel for schedule(dynamic)
    for(int i = start_i; i < (int)max_y; i++){
        for(int j = start_j; j < (int)max_x; j++){
            if((*resImg)(i, j) == 0){
                if (i+1 < max_y && copyRes(i+1,j) != 0){ // && i-1 >=fmax(min_y,0) && j+1 < max_x && j-1 >=fmax(min_x,0) ){
                    (*resImg)(i, j) = copyRes(i+1,j);
                }else if(i-1 >= fmax(min_y,0) && copyRes(i-1,j) != 0){
                    (*resImg)(i, j) = copyRes(i-1,j);
                }else if(j+1 < max_x && copyRes(i,j+1) != 0){
                    (*resImg)(i, j) = copyRes(i,j+1);
                }else if(j-1 >=fmax(min_x,0) && copyRes(i,j-1) != 0){
                    (*resImg)(i,j) = copyRes(i,j-1);
                }else if(i+1 < max_y && j+1 < max_x && copyRes(i+1,j+1)){
                    (*resImg)(i,j) = copyRes(i+1,j+1);
                }else if(i-1 >= fmax(min_y,0) && j+1 < max_x && copyRes(i-1,j+1)){
                    (*resImg)(i,j) = copyRes(i-1,j+1);
                }else if(i+1 < max_y && j-1 >=fmax(min_x,0) && copyRes(i+1,j-1)){
                    (*resImg)(i,j) = copyRes(i+1,j-1);
                }else if(i-1 >= fmax(min_y,0) && j-1 >=fmax(min_x,0) && copyRes(i-1,j-1)){
                    (*resImg)(i,j) = copyRes(i-1,j-1);
                }
            }
        }
    }
}