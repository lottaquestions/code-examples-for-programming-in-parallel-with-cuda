__global__ void partialReduceResidual(const int entries, datafloat *u, datafloat *newu, datafloat *blocksum){
    __shared__ datafloat s_blocksum[BDIM];
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    s_blocksum[threadIdx.x] = 0;

    if(id < entries){
        const datafloat diff = u[id] - newu[id];
        s_blocksum[threadIdx.x] = diff * diff;
    }

    int alive = blockDim.x;
    int t = threadIdx.x;

    while (alive > 1){
        __syncthreads(); // barrier (make sure s_blocksum is ready i.e accesses to it are completed by other threads)

        alive /= 2; // reduce active threads
        if(t < alive) s_blocksum[t] += s_blocksun[t + alive];
    }

    if(t == 0)
      blocksum[blockIdx.x] = s_blocksum[0];
}