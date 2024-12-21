#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Implements 1D thread-linear addressing
// To compile: nvcc -G -o grid3D_linear.bin grid3D_linear.cu

__device__ int a[256][512][512];  // file scope
__device__ float b[256][512][512];  // file scope


__global__ void grid3D_linear (int nx, int ny, int nz, int id){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int array_size = nx * ny * nz;
    int total_threads = gridDim.x * blockDim.x;
    int tid_start = tid;
    int pass = 0;
    __syncthreads();

    while (tid < array_size){ // "Convert linear addressing to 3D => (x y z)"
        int x = tid % nx ; // Most varying coordinate
        int y = (tid/nx) % ny;
        int z = tid / (nx * ny); // Least varying coordinate
        // Note: The division and modulus operators are expensive but can be replaced with masking and shifting operations
        // if nx and ny are known powers of 2

        /*
           Modulo operation (x % n) can be replaced by x & (n - 1), where n is a power of 2.
           Division by powers of 2 (x / n) can be replaced by x >> k, where k is the number of bits corresponding to n (i.e., n = 2^k).
         */

        // Do the work
        a[z][y][x] = tid;
        b[z][y][x] = sqrtf((float)a[z][y][x]); 
        if(tid == id){
            __syncthreads();
            printf("array size %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
            printf("thread block %3d\n", blockDim.x);
            printf("thread grid %d\n", gridDim.x);
            printf("total number of threads in grid %d\n", total_threads);
            printf("a[%d] [%d] [%d] = %i and b[%d] [%d] [%d] = %f\n", z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
            printf("rank_in_block = % d rank_in_grid = %d pass %d tid offset %d\n", threadIdx.x, tid_start, pass, tid - tid_start);
        }
        tid += gridDim.x * blockDim.x;
        pass++;
        __syncthreads();
    }
}

int main(int argc, char *argv[]){
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    int blocks = (argc > 2) ? atoi(argv[2]) : 288;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024);  // Increase the printf buffer size
    grid3D_linear<<<blocks, threads>>>(512, 512, 256, id);
    return 0;
}