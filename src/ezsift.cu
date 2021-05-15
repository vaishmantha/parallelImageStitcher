#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "ezsift.h"
#include "common.h"
#include "vvector.h"
#include "image.h"
#include "CycleTimer.h"

namespace ezsift {

void printCudaInfo();

std::vector<std::vector<std::vector<Image<float>>>> pyramids_gpu(std::vector<std::vector<Image<unsigned char>>> all_octaves, 
                                                    std::vector<int> all_nOctaves, 
                                                    int nGpyrLayers, int nDogLayers, int nLayers) {
    
    cudaDeviceSynchronize();
    
    // std::cout << "Noctaves " << nOctaves << " nGpyrLayers " << nGpyrLayers  << std::endl;
    // std::cout << "Running this from ezsift" << std::endl;

    return 0;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}

} // end namespace ezsift