#include "cooperative_groups.h"
#include "cx.h"

namespace cg = cooperative_groups;

template <int T> __device__ void show_tile(const char *tag, cg::thread_block_tile<T> p){
    int rank = p.thread_rank(); // Thread rank in the tile
    int size = p.size();  // Number of threads in tile
    int mrank = p.meta_group_rank(); // Rank of tile in parent
    int msize = p.meta_group_size(); // Number of tiles in parent

    printf("%s rank in tile %2d size %2d rank %3d num %3d net size %d\n", tag, rank, size, mrank, msize, msize*size);
}

__global__ void cgwarp(int id){
    auto grid = cg::this_grid(); // Standard cg definitions 
    auto block = cg::this_thread_block();
    auto warp32 = cg::tiled_partition<32>(block); // 32 thread warps
    auto warp16 = cg::tiled_partition<16>(block); // 16 thread tiles
    auto warp8  = cg::tiled_partition<8>(block); // 8 thread tiles
    auto tile8 = cg::tiled_partition<8>(warp32); // 8 thread sub-warps.
    auto tile4 = cg::tiled_partition<4>(tile8); // 4  thread sub-sub-warps.

    if(grid.thread_rank() == id){
        printf("warps and subwarps for thread %d:\n", id);
        show_tile<32>("warp32", warp32);
        show_tile<16>("warp16", warp16);
        show_tile<8>("warp8", warp8);
        show_tile<8>("tile8", tile8);
        show_tile<4>("tile4", tile4);
    }
}

int main(int argc, char *argv[]){
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    int blocks = (argc > 2) ? atoi(argv[2]) : 28800;
    int threads = (argc > 3) ? atoi(argv[3]) : 256;
    cgwarp<<<blocks,threads>>>(id);
    return 0;
}