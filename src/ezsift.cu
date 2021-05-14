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

double build_gaussian_pyramid_gpu(std::vector<Image<unsigned char>> &octaves,
                                std::vector<Image<float>> &gpyr, int nOctaves,
                                int nGpyrLayers) {
    int *device_input;
    cudaMalloc((void **)&device_input, 2 * sizeof(int));
    std::cout << "Running this from ezsift" << std::endl;

    return 0;
}

} // end namespace ezsift