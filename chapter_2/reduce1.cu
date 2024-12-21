#include "cx.h"
#include "cxtimers.h"
#include <random>


// To compile: nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o reduce1.bin reduce1.cu

__global__ void reduce1(float *x, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float tsum = 0.0f;
    for (int k=tid; k < N ; k += gridDim.x * blockDim.x)
        tsum += x[k];
    x[tid] = tsum; // Store partial sums in first 
                   // gridDim.x * blockDim.x elements of x
}

int main(int argc, char *argv[]){
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // 2^24
    int blocks  = (argc > 2) ? atoi(argv[2]) : 288;
	int threads = (argc > 3) ? atoi(argv[3]) : 256;

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

    // GPU reduce
    tim.reset();

    reduce1<<<blocks, threads>>>(dev_x.data().get(), N);
    reduce1<<<1, threads>>>(dev_x.data().get(), blocks * threads);
    reduce1<<<1,1>>>(dev_x.data().get(), threads);
    cudaDeviceSynchronize(); // Causes host to wait for all pending GPU operations to complete before continuing.
    double t2 = tim.lap_ms();

    double gpu_sum = dev_x[0]; // Device to host copy (1 word)
    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1, gpu_sum, t2);
    return 0;
}