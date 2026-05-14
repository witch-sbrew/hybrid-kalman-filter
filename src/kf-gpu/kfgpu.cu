#include <cuda_runtime.h>
#include "common.hpp"
#include "include/gpuint.hpp"

__global__ void run_kalman_kernel(
    const double* initial_states,
    const double* measurements,
    double* trajectories,
    int64_t filter_count,
    int64_t step_count,
    int64_t state_dim) {
  const int64_t linear_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_threads = filter_count * state_dim;
  if (linear_index >= total_threads) {
    return;
  }

  const int64_t filter_index = linear_index / state_dim;
  const int64_t state_index = linear_index % state_dim;

  double state = initial_states[filter_index * state_dim + state_index];
  double covariance = kfgpu::kInitialCovariance;

  for (int64_t step_index = 0; step_index < step_count; ++step_index) {
    covariance += kfgpu::kProcessNoise;

    const size_t measurement_index =
        static_cast<size_t>((filter_index * step_count + step_index) * state_dim + state_index);
    const double measurement = measurements[measurement_index];
    const double gain = covariance / (covariance + kfgpu::kMeasurementNoise);

    state += gain * (measurement - state);
    covariance =
        (1.0 - gain) * (1.0 - gain) * covariance +
        gain * gain * kfgpu::kMeasurementNoise;

    const size_t output_index =
        static_cast<size_t>((filter_index * step_count + step_index) * state_dim + state_index);
    trajectories[output_index] = state;
  }
}
namespace kfgpuI {
kfgpuI::GpuDBuffers alloc_dbuffers(const kfgpuI::GpuSlice& slice) {
    kfgpuI::GpuDBuffers bufs;
    bufs.initial_s = slice.filter_count 
                            * slice.state_dim * sizeof(double);
    bufs.measurement_s = slice.filter_count * slice.step_count
                             * slice.state_dim * sizeof(double);
    bufs.trajectory_s = slice.filter_count * slice.step_count
                             * slice.state_dim * sizeof(double);

    cudaMalloc(&bufs.initial_states, bufs.initial_s);
    cudaMalloc(&bufs.measurements,   bufs.measurement_s);
    cudaMalloc(&bufs.trajectories,   bufs.trajectory_s);
    return bufs;
}

void free_dbuffers(kfgpuI::GpuDBuffers& bufs) {
    cudaFree(bufs.initial_states);
    cudaFree(bufs.measurements);
    cudaFree(bufs.trajectories);
    bufs = {};
}

void kf_launch_gpu(const kfgpuI::GpuSlice& slice, kfgpuI::GpuDBuffers& bufs, cudaStream_t stream){
    cudaMemcpyAsync(bufs.initial_states, slice.initial_states,
                    bufs.initial_s, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bufs.measurements, slice.measurements,
                    bufs.measurement_s, cudaMemcpyHostToDevice, stream);

    //copied from main fn
    const int threads_per_block = 256;
    const int64_t total_threads = slice.filter_count * slice.state_dim;
    const int blocks = static_cast<int>(
        (total_threads + threads_per_block - 1) / threads_per_block);

    run_kalman_kernel<<<blocks, threads_per_block, 0, stream>>>(
        bufs.initial_states,
        bufs.measurements,
        bufs.trajectories,
        slice.filter_count,
        slice.step_count,
        slice.state_dim);

    cudaMemcpyAsync(slice.output, bufs.trajectories,
                    bufs.trajectory_s, cudaMemcpyDeviceToHost, stream);
    }
}