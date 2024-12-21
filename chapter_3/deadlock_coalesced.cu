#include "cooperative_groups.h"

namespace cg = cooperative_groups;

__global__ deadlock_coalesced(int gsync, int dolock){
    __shared__ int lock;
    if(threadIdx.x == 0) lock = 0;
    __syncthreads(); // Normal syncthreads

    if(threadIdx.x < gsync){ // Group A
        auto a = cg::coalesced_threads(); // Has all the functionality of a 32-thread tiled_partition
        // representing the local warp but adds a hidden bitmask to all member functions selecting
        // just the active threads.
        a.sync(); // Sync A

        // Deadlock unless we set lock to 1
        if(threadIdx.x == 0) lock = 1;
    } else if (threadIdx.x < 2 * gsync) { // Group B
        auto a = cg::coalesced_threads();
        a.sync(); // Sync B
    }

    // Group C no longer in deadlock danger with use of coalesced_group synchronization
    if(dolock) while (lock != 1);

    // See message only if no deadlock
    if(threadIdx.x == 0 && blockIdx.x == 0)
        printf("Deadlock coalesced OK \n");
}