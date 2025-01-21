
#include "cx.h"
#include "cxtimers.h"
#include "cxbinio.h"

// To compile nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o stencil2d_shared_memory.bin stencil2d_shared_memory.cu

// 2-D stencil for Laplace's equation
/*
                  | 0   1  0 |
  solves grad f = | 1  -4  1 | u= 0
                  | 0   1  0 |
  which is u_-1,0 + u_1,0 + u_0,-1 + u_0,1 - 4 u_0,0 = 0

  Moving U_0,0 to the right and then simplifying gives
  u_0,0 = (u_-1,0 + u_1,0 + u_0,-1 + u_0,1) / 4
  which is what the 2-D stencil does.

  This version performs tiling in shared memory. I should analyze the performance of the kernel to see how
  it does compared to the version that did not use shared memory. The expectation
  is that this kernel will be slower due to the cost of setup each shared memory tile with halo.

  Buffers are swapped in a ping pong fashion to allow iteration and hence convergence to a 
  solution. The method is called double buffering.
  For boundary conditions:
      - Left and right hand columns are set to 1.
      - Top and bottom rows are set to zero.
*/

template< int Nx, int Ny>
__global__ void stencil2D_sm(cr_Ptr<float> input, r_Ptr<float> output, int nx, int ny){
    // Tile includes a 1 element halo
    __shared__ float tile[Ny][Nx];

    auto idx = [&nx](int y, int x){ return y*nx + x; };

    // Tiles overlap hence y0 and x0 strides reduced by twice the halo width
    int x0 = (blockDim.x - 2) * blockIdx.x; // x origin
    int y0 = (blockDim.y - 2) * blockIdx.y; // y origin
    int xa = x0 + threadIdx.x; // thred x in array
    int ya = y0 + threadIdx.y; // thread y in array
    int xTile = threadIdx.x; // thread x in tile
    int yTile = threadIdx.y; // thread y in tile

    // Out of range checks
    if (xa >= nx || ny >= ny){
        return;
    }

    // Fill shared memory (tile) with halo of 1 element width. Share memory
    // will have (Ny-2) * (Nx-2) active points
    tile[yTile][xTile] = input[idx(ya, xa)];
    __syncthreads();

    // Check that we are inside the array
    if(xa < 1 || xa > (nx -1) || ya < 1 || ya > (ny -1)){
        return;
    }

    // Check that we are inside the tile, excluding the 1 element halo.
    if(xTile < 1 || xTile > (Nx -1) || yTile < 1 || yTile > (Ny-1)){
        return;
    }
    output[idx(ya, xa)] = 0.25f *(tile[yTile - 1][xTile] + tile[yTile + 1][xTile] + tile[yTile][xTile - 1] +  tile[yTile][xTile + 1] );
}

int stencil2D_host(cr_Ptr<float> a,r_Ptr<float> b,int nx,int ny)
{
	auto idx = [&nx](int y,int x){ return y*nx+x; };
	// omit edges
	for(int y=1;y<ny-1;y++){
        for(int x=1;x<nx-1;x++) {
            b[idx(y,x)] =0.25f*(a[idx(y,x+1)] + a[idx(y,x-1)] + a[idx(y+1,x)] + a[idx(y-1,x)]);
        }
    }
    
	return 0;
}


int main(int argc, char* argv[]){
    int nx =        (argc>1) ? atoi(argv[1]) : 1024;
	int ny =        (argc>2) ? atoi(argv[2]) : 1024;
	int iter_host = (argc>3) ? atoi(argv[3]) : 1000;
	int iter_gpu  = (argc>4) ? atoi(argv[4]) : 10000;
	uint threadx  = (argc>5) ? atoi(argv[5]) : 16;  // must be 32 or 16
	uint thready  = (argc>6) ? atoi(argv[6]) : 16;  // must be 32, 16 or 8 & <= threadx

    int size = nx * ny;

    thrustHvec<float> a(size);
    thrustHvec<float> b(size);
    thrustDvec<float> dev_a(size);
    thrustDvec<float> dev_b(size);

    // Solve Poisson's equation inside of a ny * nx rectangle.
    // Set edges x = 0 and x = nx -1 to 1
    auto idx = [&nx](int y, int x){ return y * nx + x;};
    for (int y = 0 ; y < ny ; y++){
        a [idx(y,0)] = a [idx(y, (nx -1))] = 1.0f;
    }

    // Corner adjustment
    a [idx(0,0)] = a [idx(0,nx -1)] = a [idx(ny-1,0)] = a [idx(ny-1,nx -1)] = 0.5f;

    // Copy to other host array
    b = a;

    // Copy to GPU memory
    dev_a = a;
    dev_b = a;

    cx::timer tim; // apply stencil iter_host times
    for (int k = 0; k < iter_host/2; k++){
        // Ping-pong the buffers
        stencil2D_host(a.data(), b.data(), nx, ny); // a => b
        stencil2D_host(b.data(), a.data(), nx, ny); // b => a
    }
    double t1 = tim.lap_ms();
	double gflops_host  = (double)(iter_host*4) * (double)size / (t1*1000000);
	cx::write_raw("stencil2Dsm_host.raw", a.data(), size);

    // Below for stencil2D_sm ========================================================================
    dim3 threads = {threadx, thready, 1};

    // Extra thread blocks needed for halos
    dim3 blocks_sm = {(nx + threads.x -1 -2)/(threads.x-2),(ny + threads.y -1 - 2)/(threads.y - 2), 1};

    tim.reset();
	if(threadx==16 && thready==16){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<16,16><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<16,16><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else if(threadx==32 && thready==32){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<32,32><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<32,32><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else if(threadx==32 && thready==16){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<32,16><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<32,16><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else if(threadx==32 && thready==8){
		for(int k=0;k<iter_gpu/2;k++){  // ping pong buffers dev_a and dev_b
			stencil2D_sm<32,8><<<blocks_sm,threads>>>(dev_a.data().get(),dev_b.data().get(),nx,ny);
			stencil2D_sm<32,8><<<blocks_sm,threads>>>(dev_b.data().get(),dev_a.data().get(),nx,ny);
		}
	}
	else {printf("bad sm config\n"); return 1;}

	cudaDeviceSynchronize();
	double t2 = tim.lap_ms();

	a = dev_a;
	//
	// do something with result
	cx::write_raw("stencil2Dsm_gpu.raw",a.data(),size);
	//
	double gflops_gpu = (double)(iter_gpu*4)*(double)size/(t2*1000000);
	double speedup = gflops_gpu/gflops_host;
	printf("stencil2d size %d x %d speedup %.3f\n",nx,ny,speedup);
	printf("host iter %8d time %9.3f ms GFlops %8.3f\n",iter_host,t1,gflops_host);
	printf("gpu  iter %8d time %9.3f ms GFlops %8.3f\n",iter_gpu,t2,gflops_gpu);

	// for logging
	FILE* flog = fopen("stencil4PTsm_gpu.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_gpu, gflops_gpu);
	fclose(flog);
	flog = fopen("stencil4PTsm_host.log", "a");
	fprintf(flog, "%4d %4d %6d %8.3f\n", nx, ny, iter_host, gflops_host);
	fclose(flog);
	return 0;
}