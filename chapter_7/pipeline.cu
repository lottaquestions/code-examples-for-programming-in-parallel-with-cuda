#include "cx.h"
#include "cxtimers.h"
#include "helper_math.h"

// To compile: nvcc -I/home/lottaquestions/nvidia-installers/cuda-samples/Common -G -o pipeline.bin pipeline.cu

__global__ void mashData(cr_Ptr<float> a, r_Ptr<float> b, uint asize, int ktime){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int k = id; k < asize; k +=stride){
        float sum = 0.0f;
        for(int m = 0; m < ktime; m++){
            sum += sqrtf(a[k]*a[k] + (float)(threadIdx.x%32) + (float)m);
        }
        b[k] = sum;
    }
}

int main(int argc, char *argv[]){
    int blocks = (argc > 1) ? atoi(argv[1]) : 256;
    int threads = (argc > 2) ? atoi(argv[2]) : 256;
    uint dsize = (argc > 3) ? 1 << atoi(argv[3]) : 1 << 28; // data size
    
    int frames = (argc > 4) ? atoi(argv[4]) : 16;
    int ktime = (argc > 5) ? atoi(argv[5]) : 60; // workload

    int maxcon = (argc > 6) ? atoi(argv[6]) : 8; // max connections
    uint fsize = dsize/frames; // frame size

    if(maxcon > 0){
        // The env var CUDA_DEVICE_MAX_CONNECTIONS sets the number of compute and copy engine concurrent connections (work queues)
        // from the host to each device of compute capability 3.5 and above.
        char set_maxconnect[256];
        sprintf(set_maxconnect, "CUDA_DEVICE_MAX_CONNECTIONS=%d", maxcon);
        putenv(set_maxconnect);
    }
    thrustHvecPin<float> host(dsize);  // host data buffer
    thrustDvec<float> dev_in(dsize);   // dev input buffer
    thrustDvec<float> dev_out(dsize);  // dev output buffer

    for (uint k = 0; k < dsize; k++){
        host[k] = (float) (k % 77) * sqrt(2.0);
    }

    // Buffer for stream objects
    thrustHvec<cudaStream_t> streams(frames);
    for(int i = 0; i < frames; i++){
        cudaStreamCreate(&streams[i]);
    }

    float *hptr = host.data(); // Copy H2D
    float *in_ptr = dev_in.data().get();
    float *out_ptr = dev_out.data().get();

    cx::timer tim;

    // Data transfers & kernel launch in each async stream
    for(int f = 0; f < frames; f++){
        if(maxcon > 0){
            // This block is for multiple async streams
            cudaMemcpyAsync(in_ptr, hptr, sizeof(float)*fsize, cudaMemcpyHostToDevice, streams[f]);
            if(ktime > 0){
                mashData<<<blocks, threads, 0, streams[f]>>>(in_ptr, out_ptr, fsize, ktime);
            }
            cudaMemcpyAsync(hptr, out_ptr, sizeof(float)*fsize, cudaMemcpyDeviceToHost, streams[f]);
        } else {
            // This block is for the single synchronous default stream
            cudaMemcpyAsync(in_ptr, hptr, sizeof(float)*fsize, cudaMemcpyHostToDevice, 0);
            if(ktime > 0)
                mashData<<<blocks,threads,0,0>>>(in_ptr, out_ptr, fsize, ktime);
            cudaMemcpyAsync(hptr, out_ptr, sizeof(float)*fsize, cudaMemcpyDeviceToHost,0);
        }
        hptr += fsize;  // point to next frame
        in_ptr += fsize;
        out_ptr += fsize;
    }
    cudaDeviceSynchronize();
    double t1 = tim.lap_ms();
    printf("time %.3f ms\n", t1);

    // Continue host calculations here
    std::atexit([]{cudaDeviceReset();});
    return 0;
    
}
