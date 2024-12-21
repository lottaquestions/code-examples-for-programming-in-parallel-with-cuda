#include "cx.h"
#include "cxtimers.h"
#include <random>


// To compile: nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o reduce0.bin reduce0.cu

__global__ void reduce0(float *x, int m){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}

int main(int argc, char *argv[]){
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // 2^24

    thrust::host_vector<float> x(N);
    thrust::device_vector<float> dev_x(N);

    // Initialize x with random numbers and copy to dx
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for (int k = 0; k < N; k++)
        x[k] = fran(gen);

    dev_x = x; // Host to device copy of N words.

    cx::timer tim;
    // Host reduce
    double host_sum = 0.0;
    for(int k = 0; k < N; k++) 
        host_sum += x[k];
    double t1 = tim.lap_ms();

    // Simple GPU reduce for N = power of 2
    tim.reset();

    for(int m = N/2; m > 0; m/=2){
        int threads = std::min(256, m);
        int blocks = std::max(m/256, 1);
        reduce0<<<blocks, threads>>>(dev_x.data().get(), m);
    }
    cudaDeviceSynchronize(); // Causes host to wait for all pending GPU operations to complete before continuing.
    double t2 = tim.lap_ms();

    double gpu_sum = dev_x[0]; // Device to host copy (1 word)
    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1, gpu_sum, t2);
    return 0;
}