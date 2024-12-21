#include "cx.h"
#include "cxtimers.h"
#include <random>


// To compile: nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o reduce3.bin reduce3.cu

__global__ void reduce3(float *y, float *x, int N){
    extern __shared__ float tsum[]; // Dynamic shared memory
    int id = threadIdx.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    tsum[id] = 0.0f;
    for(int k = tid ; k < N ; k += stride) tsum[id] += x[k];
    __syncthreads();

    // Next higher power of 2
    int block2 = cx::pow2ceil(blockDim.x);

    // Power of 2 reduction loop
    for(int k = block2/2 ; k > 0; k /=2){
        if(id < k && (id + k) < blockDim.x)  tsum[id] += tsum[id + k];
        __syncthreads();
    }
    // Store one value per thread block
    if (id == 0) y[blockIdx.x] = tsum[0];
}

int main(int argc, char *argv[]){
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24;
    int blocks = (argc > 2) ? atoi(argv[2]) : 256;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;

    thrust::host_vector<float> x(N);
    thrust::device_vector<float> dx(N);
    thrust::device_vector<float> dy(blocks);

    // Initialize x with random numbers
    std::default_random_engine gen(12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k = 0; k < N ; k++) x[k] = fran(gen);
    dx = x; // Host to device copy (N words)

    cx::timer tim;
    // Host reduce
    double host_sum = 0.0;
    for(int k = 0; k < N; k++) 
        host_sum += x[k];
    double t1 = tim.lap_ms();

    // Simple GPU reduce for any value of N
    tim.reset();
    reduce3<<<blocks, threads, threads*sizeof(float)>>>(dy.data().get(), dx.data().get(), N);
    reduce3<<<1, blocks, blocks*sizeof(float)>>>(dx.data().get(), dy.data().get(), blocks);
    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();
    double gpu_sum = dx[0]; // Device to host copy (1 word)
    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", N, host_sum, t1, gpu_sum, t2);
    return 0;

}