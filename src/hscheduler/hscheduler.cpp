#include <iostream>
#include <string>

#include <cnpy.h>
#include "hscheduler.hpp"

int main(int argc, char **argv){
    std::cout << "hello from hs" <<std::endl;

    cnpy::NpyArray x0s_npy  = cnpy::npy_load("initial_states.npy");
    cnpy::NpyArray meas_npy = cnpy::npy_load("measurements.npy");
    double* x0s_data  = x0s_npy.data<double>();
    double* meas_data = meas_npy.data<double>();

    int N = std::stoi(argv[1]), T = std::stoi(argv[2]);
    const int state_dim = 64;

    kf::PinnedVector<double> pinned_states(x0s_data, x0s_data + sizeof(x0s_data)/sizeof(double));
    kf::PinnedVector<double> pinned_meas(meas_data, meas_data + sizeof(meas_data)/sizeof(double));
    kf::PinnedVector<double> pinned_output(
        N * T * state_dim);

    kf::KFInstance job {
        .filter_count   = N,
        .step_count     = T,
        .state_dim      = state_dim,
        .initial_states = pinned_states.data(),
        .measurements   = pinned_meas.data(),
        .output         = pinned_output.data(),
    };

    kf::HScheduler hs;
    auto stats = hs.run(job);

    std::cout << "cpu=" << stats.cpu_ms << "ms  "
              << "gpu=" << stats.gpu_ms << "ms  "
              << "idle=" << stats.idle_ms << "ms\n";
              
    return 0;
}