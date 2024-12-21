#include "cx.h"
#include "cxtimers.h"
#include <random>

// To compile: nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o gputiled_matrix.bin gputiled_matrix.cu

template<int TS>
__global__ void gputiled(float * __restrict C, float * __restrict A, float * __restrict B, int Ay, int Ax, int Bx){
    __shared__ float Atile[TS][TS]; // Tile A e.g. [16][16]
    __shared__ float Btile[TS][TS]; // Tile B e.g. [16][16]
    int tx = threadIdx.x; // Tile col index j
    int ty = threadIdx.y; // Tile row index i
    int ocx = blockDim.x * blockIdx.x; // Tile x origin in C
    int ocy = blockDim.y + blockIdx.y; // Tile y origin in C

    int ax = tx; // j or x in first tile on A
    int ay = ocy + ty; // i or y in first tile on A and C
    int bx = ocx + tx; // j or x in first tile on B and C
    int by = ty;  // i or y in first tile on B

    float csum = 0.0f;
    for (int t = 0; t < gridDim.x; t++){
        Atile[ty][tx] = A[ay*Ax+ax]; // Copy A to shared memory
        Btile[ty][tx] = B[by*Bx+bx]; // Copy A to shared memory
        __syncthreads();

        for(int k = 0; k < TS; k++)
           csum += Atile[ty][k]*Btile[k][tx];
        __syncthreads();
        ax += TS; // Step A tiles along rows of A
        by += TS; // Step B tiles down cols of B
    }
    C[ay*Bx+bx] = csum;
}


int main(int argc,char *argv[])
{
	int Arow = (argc > 1) ? atoi(argv[1]) : 1 << 10; // default 2^10
	int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
	int Brow = Acol;
	int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
	int Crow = Arow;
	int Ccol = Bcol;
	uint tilex = (argc > 4) ? atoi(argv[4]) : 32;
	int nacc = (argc > 5) ? atoi(argv[5]) : 100;   // for timing

	thrust::host_vector<float>       A(Arow*Acol);
	thrust::host_vector<float>       B(Brow*Bcol);
	thrust::host_vector<float>       C(Crow*Ccol);
	thrust::device_vector<float> dev_A(Arow*Acol);
	thrust::device_vector<float> dev_B(Brow*Bcol);
	thrust::device_vector<float> dev_C(Crow*Ccol);

	// initialise x with random numbers and copy to dx.
	std::default_random_engine gen(12345678);
	std::uniform_real_distribution<float> fran(0.0,1.0);
	for(int k = 0; k<Arow*Acol; k++) A[k] = fran(gen);
	for(int k = 0; k<Brow*Bcol; k++) B[k] = fran(gen);
	dev_A = A;  // H2D copy
	dev_B = B;  // H2D copy

    dim3 threads = {tilex, tilex, 1}; // Square
    dim3 blocks = {(Bcol+threads.x-1)/threads.x, (Arow+threads.y-1)/threads.y, 1};

    cx::timer tim;
    if(tilex == 8)
        gputiled<8><<<blocks, threads>>>(dev_C.data().get(), dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    else if (tilex == 16)
        gputiled<16><<<blocks, threads>>>(dev_C.data().get(), dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    else if (tilex == 32)
        gputiled<32><<<blocks, threads>>>(dev_C.data().get(), dev_A.data().get(), dev_B.data().get(), Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    double t3 = tim.lap_ms()/(double)(nacc);
	C = dev_C; // D2H copy

	double flops = 2.0*(double)Arow*(double)Acol*(double)Bcol;
	double gflops = flops/(t3*1000000.0);
	double gbytes = gflops*6.0; // i.e 12 bytes per term
	printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.3f GBytes %.3f (gputiled)\n",Arow,Acol,Brow,Bcol,t3,gflops,gbytes);

	return 0;
}