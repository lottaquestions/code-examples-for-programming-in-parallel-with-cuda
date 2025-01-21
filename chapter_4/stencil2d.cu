#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

// nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o stencil2d.bin stencil2d.cu

// 2-D stencil for Laplace's equation
/*
                  | 0   1  0 |
  solves grad f = | 1  -4  1 | u= 0
                  | 0   1  0 |
  which is u_-1,0 + u_1,0 + u_0,-1 + u_0,1 - 4 u_0,0 = 0

  Moving U_0,0 to the right and then simplifying gives
  u_0,0 = (u_-1,0 + u_1,0 + u_0,-1 + u_0,1) / 4
  which is what the 2-D stencil does

  The CUDA kernel is launched with one thread per element of the input buffer which is a 2D grid.
  This is what the kernel expects - sufficient threads to cover all elements.
  Buffers are swapped in a ping pong fashion to allow iteration and hence convergence to a 
  solution. The method is called double buffering.
  For boundary conditions:
      - Left and right hand columns are set to 1.
      - Top and bottom rows are set to zero.
*/


__global__ void stencil2D(cr_Ptr<float> input, r_Ptr<float> output, int nx, int ny){
    // C++ ordering of suffices
    auto idx = [&nx](int y, int x) { return y * nx + x; };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Exclude edges and out of range
    if (x < 1 || y < 1 || x >= nx -1 || y >= ny -1) {
        return;
    }

    // This is a memory-bound kernel with 5 gobal memory accesses. 4 reads and 1 write.
    output[idx(y, x)] = 0.25f * (input[idx(y, x + 1)] + 
                                 input[idx(y, x - 1)] +
                                 input[idx(y + 1, x)] +
                                 input[idx(y - 1, x)]);
}

void stencil2D_host(cr_Ptr<float> input, r_Ptr<float> output, int nx, int ny){
    // C++ ordering of suffices
    auto idx = [&nx](int y, int x) { return y * nx + x; };

    // Omit edges by starting at 1 instead of 0
    for (int y = 1 ; y < ny; y++){
        for (int x = 1; x < nx; x++){
            output[idx(y, x)] = 0.25f * (input[idx(y, x + 1)] + 
                                 input[idx(y, x - 1)] +
                                 input[idx(y + 1, x)] +
                                 input[idx(y - 1, x)]);
        }
    }
}

int main(int argc, char *argv[]){
    int nx = (argc > 1) ? atoi(argv[1]) : 1024;
    int ny = (argc > 2) ? atoi(argv[2]) : 1024;
    int iter_host = (argc > 3) ? atoi(argv[3]) : 1000;
    int iter_gpu = (argc > 4) ? atoi(argv[4]) : 1000;
    int size  = nx * ny;

    thrustHvec<float> a(size);
    thrustHvec<float> b(size);
    thrustDvec<float> dev_a(size);
    thrustDvec<float> dev_b(size);

    // Set x = 0 and x = nx -1 edges to 1
    auto idx = [&nx](int y, int x){ return y * nx + x; };
    for (int y = 0; y < ny ; y++){
        a[idx(y, 0)] = a[idx(y, nx -1)] = 1.0f;
    }
    // Corner adjustment
    a[idx(0,0)] = a[idx(0, nx -1)] = a[idx(ny-1,0)] = a[idx(ny - 1, nx - 1)] = 0.5f;
    // Copy a to both device buffers
    dev_a = a;
    dev_b = a;

    cx::timer tim;
    for(int k = 0; k < iter_host/2; k++ ){
        // Ping pong buffers
        stencil2D_host(a.data(), b.data(), nx, ny); // a => b
        stencil2D_host(b.data(), a.data(), nx, ny); // b => a
    }
    double t1 = tim.lap_ms();
    double gflops_host = double(iter_host * 4) * double(size)/(t1 * 1000000);

    dim3 threads = {16, 16, 1};
    dim3 blocks = {(nx + threads.x -1)/threads.x,
                   (ny + threads.y -1)/threads.x, 1};

    tim.reset();
    for(int k = 0; k < iter_gpu/2 ; k++){
        // Ping pong buffers
        stencil2D<<<blocks,threads>>>(dev_a.data().get(), dev_b.data().get(), nx, ny); // a => b
        stencil2D<<<blocks,threads>>>(dev_b.data().get(), dev_a.data().get(), nx, ny); // a => b
    }
    cudaDeviceSynchronize();
    double t2 = tim.lap_ms();

    a = dev_a;

    double gflops_gpu = (double)(iter_gpu*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_gpu, t2,gflops_gpu);

    return 0;
}