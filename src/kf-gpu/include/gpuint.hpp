#pragma once
#include <cuda_runtime.h>

//interface to run gpu via scheduler
namespace kfgpuI {

struct GpuSlice {
    const double* initial_states;
    const double* measurements; 
    double*       output;      

    int filter_count;           
    int step_count;
    int state_dim;
};

struct GpuDBuffers {
    double* initial_states  = nullptr;
    double* measurements    = nullptr;
    double* trajectories    = nullptr;

    size_t  initial_s   = 0;
    size_t  measurement_s = 0;
    size_t  trajectory_s  = 0;
};

GpuDBuffers alloc_dbuffers(const GpuSlice& slice);
void free_dbuffers(GpuDBuffers& bufs);
void kf_launch_gpu(const kfgpuI::GpuSlice& slice, kfgpuI::GpuDBuffers& bufs, cudaStream_t stream);
}