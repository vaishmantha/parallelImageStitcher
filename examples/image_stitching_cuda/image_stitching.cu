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

// __global__ void kernelCompose(MatrixXd *resImg, MatrixXd *newImage, int start_i, int start_j, double max_x, double max_y){
//     int threadIdxX = blockIdx.x * blockDim.x + threadIdx.x; 
//     int threadIdxY = blockIdx.y * blockDim.y + threadIdx.y; 

//     int i = threadIdxX + start_i;
//     int j = threadIdxY + start_j; 
//     if(i < (int)max_y && j < (int)max_x) {
//         if ((*resImg)(i,j) == 0){
//             (*resImg)(i,j) = (*newImage)(i,j);
//         }
//         if ((*resImg)(i,j) != 0 && (*newImage)(i, j) != 0){
//             (*resImg)(i,j) = fmax((*newImage)(i,j), (*resImg)(i,j));
//         }
//     }
// }

// void placeImage(MatrixXd newImage, MatrixXd* resImg, double min_x, double min_y, double max_x, double max_y){
//     // int w = newImage.cols();
//     // int h = newImage.rows();
//     // printf("w: %d, h: %d", w, h);
//     int start_i = (int)fmax(min_y,0);
//     int start_j = (int)fmax(min_x,0);
//     MatrixXd *resImg_device; 
//     MatrixXd *newImage_device; 

//     std::cout << "Size of matrixxd " << sizeof(MatrixXd) << std::endl;
//     cudaMalloc((void **)&resImg_device, sizeof(double)*sizeof(resImg->rows()*resImg->cols()));
//     cudaMalloc((void **)&newImage_device, sizeof(double)*sizeof(newImage.rows()*newImage.cols()));
//     cudaMemcpy(resImg_device, resImg, sizeof(double)*sizeof(resImg->rows()*resImg->cols()), cudaMemcpyHostToDevice);
//     cudaMemcpy(newImage_device, &newImage, sizeof(double)*sizeof(newImage.rows()*newImage.cols()), cudaMemcpyHostToDevice);
    
//     dim3 blockDim(16, 16, 1);
//     dim3 gridDim((((int)max_y) - start_i + blockDim.x - 1) / blockDim.x, (((int)max_x) - start_j + blockDim.y - 1) / blockDim.y);
//     kernelCompose<<<gridDim, blockDim>>>(resImg_device, newImage_device, start_i, start_j, max_x, max_y); 
//     cudaMemcpy(resImg, resImg_device, sizeof(double)*sizeof(resImg->rows()*resImg->cols()), cudaMemcpyDeviceToHost);
//     // std::cout << *resImg << std::endl;
//     // #pragma omp parallel for schedule(dynamic)
//     // for (int i = start_i; i < (int)max_y; i++){ //access as row col
//     //     for (int j = start_j; j < (int)max_x; j++){
//     //         if ((*resImg)(i,j) == 0){
//     //             (*resImg)(i,j) = newImage(i,j);
//     //         }
//     //         if ((*resImg)(i,j) != 0 && newImage(i, j) != 0){
//     //             (*resImg)(i,j) = fmax(newImage(i,j), (*resImg)(i,j));
//     //         }
//     //     }
//     // }
    
    
//     MatrixXd copyRes = (*resImg);
//     // dim3 blockDim(16, 16, 1);
//     // dim3 gridDim((((int)max_y) - start_i + blockDim.x - 1) / blockDim.x, (((int)max_x) - start_j + blockDim.y - 1) / blockDim.y);
//     // MatrixXd *resImg_device; 
//     // MatrixXd *copyRes_device; 

//     // cudaMalloc((void **)&resImg_device, sizeof(MatrixXd));
//     // cudaMalloc((void **)&copyRes_device, sizeof(MatrixXd));
//     // cudaMemcpy(resImg_device, resImg, sizeof(MatrixXd), cudaMemcpyHostToDevice);
//     // cudaMemcpy(copyRes_device, &copyRes, sizeof(MatrixXd), cudaMemcpyHostToDevice);
    
//     // kernelInterpolate<<<gridDim, blockDim>>>(resImg_device, copyRes_device, start_i, start_j, max_x, max_y);

//     // cudaMemcpy(resImg, resImg_device, sizeof(MatrixXd), cudaMemcpyDeviceToHost);

//     #pragma omp parallel for schedule(dynamic)
//     for(int i = start_i; i < (int)max_y; i++){
//         for(int j = start_j; j < (int)max_x; j++){
//             if((*resImg)(i, j) == 0){
//                 if (i+1 < max_y && copyRes(i+1,j) != 0){ // && i-1 >=fmax(min_y,0) && j+1 < max_x && j-1 >=fmax(min_x,0) ){
//                     (*resImg)(i, j) = copyRes(i+1,j);
//                 }else if(i-1 >= fmax(min_y,0) && copyRes(i-1,j) != 0){
//                     (*resImg)(i, j) = copyRes(i-1,j);
//                 }else if(j+1 < max_x && copyRes(i,j+1) != 0){
//                     (*resImg)(i, j) = copyRes(i,j+1);
//                 }else if(j-1 >=fmax(min_x,0) && copyRes(i,j-1) != 0){
//                     (*resImg)(i,j) = copyRes(i,j-1);
//                 }else if(i+1 < max_y && j+1 < max_x && copyRes(i+1,j+1)){
//                     (*resImg)(i,j) = copyRes(i+1,j+1);
//                 }else if(i-1 >= fmax(min_y,0) && j+1 < max_x && copyRes(i-1,j+1)){
//                     (*resImg)(i,j) = copyRes(i-1,j+1);
//                 }else if(i+1 < max_y && j-1 >=fmax(min_x,0) && copyRes(i+1,j-1)){
//                     (*resImg)(i,j) = copyRes(i+1,j-1);
//                 }else if(i-1 >= fmax(min_y,0) && j-1 >=fmax(min_x,0) && copyRes(i-1,j-1)){
//                     (*resImg)(i,j) = copyRes(i-1,j-1);
//                 }
//             }
//         }
//     }
// }
__global__ void kernelWarpPerspective(double* H, int png_width, int png_height, int newImCols, int newImRows, unsigned char* out_r_device, 
                                    unsigned char* out_g_device, unsigned char* out_b_device, unsigned char* out_a_device, unsigned char* png_r, unsigned char* png_g,
                                    unsigned char* png_b, unsigned char* png_a){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i > png_height && j > png_width)
        return;
    
    int tmp00 = j;
    int tmp10 = i;
    int tmp20 = 1;

    double prod_00 = H[0]*tmp00 + H[1]*tmp10 + H[2]*tmp20;
    double prod_10 = H[3]*tmp00 + H[4]*tmp10 + H[5]*tmp20;
    double prod_20 = H[6]*tmp00 + H[7]*tmp10 + H[8]*tmp20;
    
    int res_00 = (int)prod_00/prod_20;
    int res_10 = (int)prod_10/prod_20;
    if(res_00 >= 0 && res_00 < newImCols && res_10 >= 0 && res_10 < newImRows){
        out_r_device[res_10*newImCols+res_10] = (int)png_r[i*png_width + j]; //try flipped too
        out_g_device[res_10*newImCols+res_10] = (int)png_g[i*png_width + j]; //try flipped too
        out_b_device[res_10*newImCols+res_10] = (int)png_b[i*png_width + j]; //try flipped too
        out_a_device[res_10*newImCols+res_10] = (int)png_a[i*png_width + j]; //try flipped too
    }
    
}

void warpPerspective(unsigned char* png_r, unsigned char* png_g, unsigned char* png_b, unsigned char* png_a, 
int png_width, int png_height, MatrixXd* newImR,MatrixXd* newImG,MatrixXd* newImB, MatrixXd* newImA, MatrixXd H){
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((png_width + blockDim.x - 1) / blockDim.x, ((png_height + blockDim.y - 1) / blockDim.y));

    double* H_device;
    double *H_data = H.data();
    cudaMalloc((void **)&H_device, 3*3*sizeof(double)); //homography
    cudaMemcpy(H_device, H_data, 3*3*sizeof(double), cudaMemcpyHostToDevice);

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

    unsigned char* out_r_device;
    unsigned char* out_g_device;
    unsigned char* out_b_device;
    unsigned char* out_a_device;
    cudaMalloc((void **)&out_r_device, newImR->rows()*newImR->cols()*sizeof(unsigned char)); //try int as well
    cudaMalloc((void **)&out_g_device, newImG->rows()*newImG->cols()*sizeof(unsigned char));
    cudaMalloc((void **)&out_b_device, newImB->rows()*newImB->cols()*sizeof(unsigned char));
    cudaMalloc((void **)&out_a_device, newImA->rows()*newImA->cols()*sizeof(unsigned char));
    
    kernelWarpPerspective<<<gridDim, blockDim>>>(H_device, png_width, png_height, newImR->cols(), newImG->rows(), 
                                                out_r_device, out_g_device, out_b_device, out_a_device, png_r_device, png_g_device,
                                                png_b_device, png_a_device);

    cudaFree(H_device);
    cudaFree(png_r_device);
    cudaFree(png_g_device);
    cudaFree(png_b_device);
    cudaFree(png_a_device);

    //May not have to malloc here
    unsigned char* out_r_host = (unsigned char*)malloc(newImR->rows()*newImR->cols()*sizeof(unsigned char));
    unsigned char* out_g_host = (unsigned char*)malloc(newImR->rows()*newImR->cols()*sizeof(unsigned char));
    unsigned char* out_b_host = (unsigned char*)malloc(newImR->rows()*newImR->cols()*sizeof(unsigned char));
    unsigned char* out_a_host = (unsigned char*)malloc(newImR->rows()*newImR->cols()*sizeof(unsigned char));

    cudaMemcpy(out_r_device, out_r_host, png_height*png_width*sizeof(char), cudaMemcpyDeviceToHost); //CHECK ORDER OF ARGS HERE
    cudaMemcpy(out_g_device, out_g_host, png_height*png_width*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b_device, out_b_host, png_height*png_width*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_a_device, out_a_host, png_height*png_width*sizeof(char), cudaMemcpyDeviceToHost);

    *newImR = Eigen::Map<MatrixXd>(out_r_device);
    *newImG = Eigen::Map<MatrixXd>(out_g_device);
    *newImB = Eigen::Map<MatrixXd>(out_b_device);
    *newImA = Eigen::Map<MatrixXd>(out_a_device);

    cudaFree(out_r_device);
    cudaFree(out_g_device);
    cudaFree(out_b_device);
    cudaFree(out_a_device);
    free(out_r_host);
    free(out_g_host);
    free(out_b_host);
    free(out_a_host);


    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}

// void warpPerspective(unsigned char* png_r, unsigned char* png_g, unsigned char* png_b, unsigned char* png_a, 
//         int png_width, int png_height, MatrixXd* newImR,MatrixXd* newImG,MatrixXd* newImB, MatrixXd* newImA, MatrixXd H){
//     //FIX: Need to create matrix of form Nx3 and do the matrix multiply all at once- cuda kernel
//     int i; 
//     #pragma omp parallel for collapse(2)
//     for(i=0; i< png_height; i++){ 
//         for(int j=0; j<png_width; j++){
//             MatrixXd tmp = MatrixXd::Constant(3,1, 0.0);
//             tmp(0,0) = j;
//             tmp(1,0) = i;
//             tmp(2,0) = 1;
//             MatrixXd res = H*tmp;
//             MatrixXd tm =  Matslice(res, 2, 0, 1, res.cols()).replicate(3,1); //(MatrixXd array, int start_row, int start_col, int height, int width)
//             res = res.cwiseQuotient(tm);
//             if ((int)res(0,0) >= 0 && (int)res(0,0) < (*newImR).cols() && (int)res(1,0) >= 0 && (int)res(1,0) < (*newImR).rows()){
//                 (*newImR)((int)res(1,0), (int)res(0,0)) = (int)png_r[i*png_width + j];
//                 (*newImG)((int)res(1,0), (int)res(0,0)) = (int)png_g[i*png_width + j];
//                 (*newImB)((int)res(1,0), (int)res(0,0)) = (int)png_b[i*png_width + j];
//                 (*newImA)((int)res(1,0), (int)res(0,0)) = (int)png_a[i*png_width + j];
//             }
//         }
//     }
// }

