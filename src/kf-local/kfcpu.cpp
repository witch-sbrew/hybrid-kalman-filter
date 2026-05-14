#include <iostream>
#include <vector>
#include <omp.h>
#include <memory>

#include "kalman_filter/kalman_filter.h"
#include "types.h"
#include "include/kfcpu.hpp"

const int STATE_DIM = 64;

void kf_launch_cpu(kfcpu::CpuSlice& slice){
    int N = slice.filter_count;
    int T_STEPS = slice.step_count;
    const int DIM_X = STATE_DIM, DIM_Z = STATE_DIM;

    const kf::Matrix<DIM_X, DIM_X> F = kf::Matrix<DIM_X, DIM_X>::Identity();
    const kf::Matrix<DIM_X, DIM_X> Q = kf::Matrix<DIM_X, DIM_X>::Identity() * 0.01F;
    const kf::Matrix<DIM_Z, DIM_X> H = kf::Matrix<DIM_Z, DIM_X>::Identity();
    const kf::Matrix<DIM_Z, DIM_Z> R = kf::Matrix<DIM_Z, DIM_Z>::Identity() * 5.0F;
    const kf::Matrix<DIM_X, DIM_X> P0 = kf::Matrix<DIM_X, DIM_X>::Identity() * 500.0F;

    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
        std::cout << "Using " << omp_get_num_threads() << " thread\n";
        std::unique_ptr<kf::KalmanFilter<DIM_X, DIM_Z>> filter = std::make_unique<kf::KalmanFilter<DIM_X, DIM_Z>>();

        for (size_t i = 0; i < DIM_X; ++i)
            filter->vecX()(i) = static_cast<kf::float32_t>(slice.x0[n * DIM_X + i]);

        filter->matP() = P0;

        for (size_t t = 0; t < T_STEPS; ++t){
            filter->predictLKF(F, Q);

            kf::Vector<DIM_Z> vecZ;
            for (size_t i = 0; i < DIM_Z; ++i)
                vecZ(i) = static_cast<kf::float32_t>(
                    slice.m0[(n * T_STEPS + t) * DIM_Z + i]);

            filter->correctLKF(vecZ, R, H);

            for (size_t i = 0; i < DIM_X; ++i)
                slice.output[(n * T_STEPS + t) * DIM_X + i] =
                    static_cast<float>(filter->vecX()(i));
        }
    }
}