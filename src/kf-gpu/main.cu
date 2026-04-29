#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

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

}  // namespace

int main(int argc, char** argv) {
  try {
    const kfgpu::Paths paths = kfgpu::parse_paths(argc, argv);
    const kfgpu::ExperimentData experiment =
        kfgpu::load_experiment(paths.initial_states, paths.measurements);

    const size_t initial_bytes = experiment.initial_states.size() * sizeof(double);
    const size_t measurement_bytes = experiment.measurements.size() * sizeof(double);
    std::vector<double> host_trajectories(static_cast<size_t>(
        experiment.filter_count * experiment.step_count * experiment.state_dim));
    const size_t trajectory_bytes = host_trajectories.size() * sizeof(double);

    double* device_initial_states = nullptr;
    double* device_measurements = nullptr;
    double* device_trajectories = nullptr;

    check_cuda(cudaMalloc(&device_initial_states, initial_bytes), "cudaMalloc(initial_states)");
    check_cuda(cudaMalloc(&device_measurements, measurement_bytes), "cudaMalloc(measurements)");
    check_cuda(cudaMalloc(&device_trajectories, trajectory_bytes), "cudaMalloc(trajectories)");

    check_cuda(
        cudaMemcpy(
            device_initial_states,
            experiment.initial_states.data(),
            initial_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(initial_states)");
    check_cuda(
        cudaMemcpy(
            device_measurements,
            experiment.measurements.data(),
            measurement_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy(measurements)");

    const int threads_per_block = 256;
    const int64_t total_threads = experiment.filter_count * experiment.state_dim;
    const int blocks =
        static_cast<int>((total_threads + threads_per_block - 1) / threads_per_block);

    run_kalman_kernel<<<blocks, threads_per_block>>>(
        device_initial_states,
        device_measurements,
        device_trajectories,
        experiment.filter_count,
        experiment.step_count,
        experiment.state_dim);
    check_cuda(cudaGetLastError(), "run_kalman_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(
        cudaMemcpy(
            host_trajectories.data(),
            device_trajectories,
            trajectory_bytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(trajectories)");

    const std::string output_npy = paths.output_dir + "/cuda_batch_outputs.npy";
    const std::string output_raw = paths.output_dir + "/cuda_batch_raw.bin";
    kfgpu::write_npy_f64(
        output_npy,
        {experiment.filter_count, experiment.step_count, experiment.state_dim},
        host_trajectories);
    kfgpu::write_raw_final_states(
        output_raw,
        host_trajectories,
        experiment.filter_count,
        experiment.step_count,
        experiment.state_dim);
    kfgpu::copy_reference_output(paths.reference, paths.output_dir);

    cudaFree(device_trajectories);
    cudaFree(device_measurements);
    cudaFree(device_initial_states);

    std::cout << "Saved CUDA trajectories to " << output_npy << '\n';
    std::cout << "Saved final-state raw output to " << output_raw << '\n';
    std::cout << "Copied reference output to " << paths.output_dir << "/reference_outputs.npy\n";
    std::cout << "Shape: (" << experiment.filter_count << ", "
              << experiment.step_count << ", " << experiment.state_dim << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "kalman_cuda_batch failed: " << error.what() << '\n';
    return 1;
  }
}
