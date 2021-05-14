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
    int *device_input;
    cudaMalloc((void **)&device_input, 2 * sizeof(int));
    std::cout << "Running this from ezsift" << std::endl;

    return 0;
}