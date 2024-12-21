#include "cx.h"
#include "cxtimers.h"
#include <random>

__global__ void gpumult0(float *C, const float *A, const float *B, int Ay, int Ax, int Bx){
    int tx = blockIdx.x * blockDim.x + threadIdx.x; // col j
    int ty = blockIdx.y * blockDim.y + threadIdx.y; // row i

    if (ty > Ay || tx > Bx)
        return;
    
    C[ty*Bx + tx] = 0.0;

    for (int k = 0; k < Ax; k++){
        C[ty * Bx + tx] += A[ty*Ax + k] * B[ k *Bx + tx]; // Original was A[ty*Bx + k] * B[ k *Bx + tx] . Need to figure out why the Bx in A index in orig
    }    
}

int main(int argc, char *argv[]){
    int Arow = (argc > 1) ? atoi(argv[1]) : 1024;
    int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
    int Brow = Acol;
    int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
    int Crow = Arow;
    int Ccol = Bcol;

    uint tilex = (argc > 4) ? atoi(argv[4]) : 32;
    uint tiley = (argc > 5) ? atoi(argv[5]) : 8;

    thrust::host_vector<float> A(Arow * Acol);
    thrust::host_vector<float> B(Brow * Bcol);
    thrust::host_vector<float> C(Crow * Ccol);

    thrust::device_vector<float> dev_A(Arow * Acol);
    thrust::device_vector<float> dev_B(Brow * Bcol);
    thrust::device_vector<float> dev_C(Crow * Ccol);

    // Initialize A and B with random numbers
    std::default_random_engine gen (12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k= 0; k < Arow*Acol; k++)
        A[k] = fran(gen);
    for(int k= 0; k < Brow*Bcol; k++)
        B[k] = fran(gen);

    // Host to device copy
    dev_A = A;
    dev_B = B;

    dim3 threads = {tilex, tiley, 1};
    dim3 blocks = {(Bcol + threads.x - 1)/threads.x, (Arow + threads.y - 1), 1};

    cx::timer tim;
    gpumult0<<<blocks, threads>>>(dev_C.data().get(), dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();
    C = dev_C;

    double flops = 2.0*Arow*Acol*Bcol;
    double gflops = flops / (t2 * 1000000.0);
    double gbytes = gflops * 6.0; // 12 bytes per term.
    
    printf("A %d x %d B %d x %d gpu time %.3f ms Gflops %.3f GBytes %.3f\n", Arow, Acol, Brow, Bcol, t2, gflops, gbytes);
    return 0;

}