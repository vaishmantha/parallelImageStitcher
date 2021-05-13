#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

double cudaFindPeaks() {
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
    std::cout << "Running this" << std::endl;

    return 0;
}