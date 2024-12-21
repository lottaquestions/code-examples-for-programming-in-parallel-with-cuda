#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"  // cuda basic
#include "thrust/device_vector.h"
#include "cxtimers.h"

// To compile: /home/nick/Software/gcc13.2/installation/bin/g++-13.2 -std=c++20 cpusum.cc -o cpusum.bin

// To run: LD_LIBRARY_PATH=/home/nick/Software/gcc13.2/installation/lib64 ./cpusum.bin

__host__ __device__ inline float sinsum(float x, int terms){
    float x2 = x*x;
    float term = x; // First term of series
    float sum = term; // Sum of terms so far
    

    for(int n = 1; n < terms; n++){
        term *= -x2 / (2*n*(2*n+1)); // build factorial
        sum += term;
    }
    return sum;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size){
    // Unique thread ID
    int step = blockIdx.x*blockDim.x + threadIdx.x;
    while (step < steps){
        float x = step_size * step;
        sums [step] = sinsum(x, terms); // Store sums
        step += gridDim.x * blockDim.x; // Grid size stride
    }
}

int main(int argc, char *argv[]){
    // Get command line arguments
    int steps = (argc > 1) ? atoi (argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi (argv[2]) : 1000;

    int threads =  (argc > 2) ? atoi (argv[3]) : 256;
    int blocks = (argc > 3) ? atoi (argv[4]) : (steps + threads - 1)/threads; // round up


    double pi = 3.14159265358979323;

    double step_size = pi / (steps -1); // n-1 steps

    // Allocate GPU buffer and get pointer
    thrust::device_vector<float> dsums(steps);
    float *dptr = thrust::raw_pointer_cast(&dsums[0]);

    cx::timer tim;
    gpu_sin<<<blocks,threads>>>(dptr, steps, terms, (float) step_size);
    double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
    
    double gpu_time = tim.lap_ms(); // Elapsed time

    // Trapezoidal Rule correction
    gpu_sum -= 0.5 * (sinsum(0.0, terms) + sinsum (pi, terms));
    gpu_sum *= step_size;
    std::cout << "gpu_sum = " << gpu_sum << ", steps " << steps << " terms " << terms << " time " << gpu_time << std::endl;
}