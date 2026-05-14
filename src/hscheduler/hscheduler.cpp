#include <iostream>
#include <string>
#include <chrono>

#include <cnpy.h>
#include "hscheduler.hpp"

int main(int argc, char **argv){
    std::cout << "hello from hs" <<std::endl;

    cnpy::NpyArray x0s_npy  = cnpy::npy_load("initial_states.npy");
    cnpy::NpyArray meas_npy = cnpy::npy_load("measurements.npy");
    double* x0s_data  = x0s_npy.data<double>();
    double* meas_data = meas_npy.data<double>();

    int N = std::stoi(argv[1]), T = std::stoi(argv[2]);
    bool gpu_only = std::stoi(argv[3]);

    const int state_dim = 64;

    auto wall_start1 = std::chrono::high_resolution_clock::now();
    kf::PinnedVector<double> pinned_states(x0s_data, x0s_data + state_dim*N);
    kf::PinnedVector<double> pinned_meas(meas_data, meas_data + state_dim*N);
    kf::PinnedVector<double> pinned_output(
        N * state_dim);

    auto wall_end1 = std::chrono::high_resolution_clock::now();
    float wall_ms1 = std::chrono::duration<float, std::milli>(
                            wall_end1 - wall_start1).count();
    std::cout << "Alloc time: " << wall_ms1 << std::endl;

    kf::KFInstance job {
        .filter_count   = N,
        .step_count     = T,
        .state_dim      = state_dim,
        .initial_states = pinned_states.data(),
        .measurements   = pinned_meas.data(),
        .output         = pinned_output.data(),
    };

    kf::HScheduler hs(gpu_only);

    auto stats = hs.run(job);

    std::cout << "cpu=" << stats.cpu_ms << "ms  "
              << "gpu=" << stats.gpu_ms << "ms  "
              << "idle=" << stats.idle_ms << "ms\n";
    
//    for(int i = 0; i < state_dim; i++){
//         std::cout << job.output[i] << ",";
//     }
//     std::cout << std::endl;

    cnpy::npy_save("outputs.npy", job.output, {N, T, state_dim}, "w");

    return 0;
}