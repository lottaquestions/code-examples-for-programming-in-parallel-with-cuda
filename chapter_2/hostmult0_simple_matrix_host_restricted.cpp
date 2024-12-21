#include "thrust/host_vector.h"
#include "cxtimers.h"
#include <random>

// nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o hostmult0_simple_matrix_host_restricted.bin hostmult0_simple_matrix_host_restricted.cpp

int hostmult0(float * __restrict C, float * __restrict A, float * __restrict B, int Ay, int Ax, int Bx){
    // Compute C = A * B for matrices (assumes Ax = By)
    for(int i = 0; i < Ay; i++){ // every row in A
        for (int j = 0; j < Bx; j++){ // every column in B
            C[i* Bx + j] = 0.0; // row.col dot product
            for(int k = 0; k < Ax; k++){
                C[i * Bx + j] += A[i*Ax + k] * B[k * Bx + j]; // every row of current col in B hence B depends on inner counter k
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[]){
    int Arow = (argc > 1) ? atoi(argv[1]) : 1024;
    int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
    int Brow = Acol;
    int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
    int Crow = Arow;
    int Ccol = Bcol;

    thrust::host_vector<float> A(Arow*Acol);
    thrust::host_vector<float> B(Brow*Bcol);
    thrust::host_vector<float> C(Crow*Ccol);

    // Initialize A and B with random numbers
    std::default_random_engine gen (12345678);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for(int k= 0; k < Arow*Acol; k++)
        A[k] = fran(gen);
    for(int k= 0; k < Brow*Bcol; k++)
        B[k] = fran(gen);

    cx::timer tim;
    hostmult0(C.data(), A.data(), B.data(), Arow, Acol, Bcol);
    double t1 = tim.lap_ms();
    double flops = 2.0 * double(Arow) * double(Acol) * double(Bcol);
    double gflops = flops/(t1 * 1000000.0);
    double gbytes = gflops * 6.0; // 12 bytes per term
    printf("A %d x %d B %d x %d host time %.3f ms Gflops/sec %.3f\n", Arow, Acol, Brow, Bcol, t1, gflops);
    return 0;
}