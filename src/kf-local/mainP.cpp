#include <iostream>
#include <vector>
#include <omp.h>
#include <cnpy.h>
#include <chrono>
#include <memory>

#include "kalman_filter/kalman_filter.h"
#include "types.h"

static constexpr size_t N       = 1;
static constexpr size_t T_STEPS = 128;
static constexpr size_t DIM_X   = 64;
static constexpr size_t DIM_Z   = 64;

int main(){
    //Eigen::setNbThreads(2);
    std::cout << "Using " << omp_get_num_threads() << " thread\n";

    cnpy::NpyArray x0s_npy  = cnpy::npy_load("initial_states.npy");
    cnpy::NpyArray meas_npy = cnpy::npy_load("measurements.npy");

    double* x0s_data  = x0s_npy.data<double>();
    double* meas_data = meas_npy.data<double>();

    const kf::Matrix<DIM_X, DIM_X> F = kf::Matrix<DIM_X, DIM_X>::Identity();
    const kf::Matrix<DIM_X, DIM_X> Q = kf::Matrix<DIM_X, DIM_X>::Identity() * 0.01F;
    const kf::Matrix<DIM_Z, DIM_X> H = kf::Matrix<DIM_Z, DIM_X>::Identity();
    const kf::Matrix<DIM_Z, DIM_Z> R = kf::Matrix<DIM_Z, DIM_Z>::Identity() * 5.0F;
    const kf::Matrix<DIM_X, DIM_X> P0 = kf::Matrix<DIM_X, DIM_X>::Identity() * 500.0F;

    std::vector<float> results(N * T_STEPS * DIM_X);

    auto start = std::chrono::steady_clock::now();

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        std::cout << "Using " << omp_get_num_threads() << " thread\n";
        std::unique_ptr<kf::KalmanFilter<DIM_X, DIM_Z>> filter = std::make_unique<kf::KalmanFilter<DIM_X, DIM_Z>>();

        for (size_t i = 0; i < DIM_X; ++i)
            filter->vecX()(i) = static_cast<kf::float32_t>(x0s_data[n * DIM_X + i]);

        filter->matP() = P0;

        for (size_t t = 0; t < T_STEPS; ++t)
        {
            filter->predictLKF(F, Q);

            kf::Vector<DIM_Z> vecZ;
            for (size_t i = 0; i < DIM_Z; ++i)
                vecZ(i) = static_cast<kf::float32_t>(
                    meas_data[(n * T_STEPS + t) * DIM_Z + i]);

            filter->correctLKF(vecZ, R, H);

            for (size_t i = 0; i < DIM_X; ++i)
                results[(n * T_STEPS + t) * DIM_X + i] =
                    static_cast<float>(filter->vecX()(i));
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cnpy::npy_save("cpp_outputs.npy", results.data(), {N, T_STEPS, DIM_X}, "w");
    std::cout << "Saved cpp_outputs.npy — time P(" 
    << N << ", " << T_STEPS << ", " << DIM_X << ") t = " << t.count() << "\n";

    return 0;
}