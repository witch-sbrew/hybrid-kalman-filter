#include <iostream>
#include <stdint.h>

#include "kalman_filter/kalman_filter.h"
#include "types.h"

static constexpr size_t DIM_X{64};
static constexpr size_t DIM_Z{32};  // observation dimension (adjust as needed)
static constexpr kf::float32_t T{1.0F};
static constexpr kf::float32_t Q_DIAG{0.1F};  // uniform diagonal noise value

static kf::KalmanFilter<DIM_X, DIM_Z> kalmanfilter;

void executePredictionStep();
void executeCorrectionStep();

int main()
{
  executePredictionStep();
  executeCorrectionStep();

  return 0;
}

void executePredictionStep()
{
  // Initialize state vector: alternating position/velocity pairs across 32 channels
  // Layout: [pos0, vel0, pos1, vel1, ..., pos31, vel31]
  kf::Vector<DIM_X> x0 = kf::Vector<DIM_X>::Zero();
  for (size_t i = 0; i < DIM_X; i += 2)
  {
    x0(i)     = 0.0F;   // position component
    x0(i + 1) = 2.0F;   // velocity component
  }
  kalmanfilter.vecX() = x0;

  kalmanfilter.matP() = kf::Matrix<DIM_X, DIM_X>::Identity() * 0.1F;

  kf::Matrix<DIM_X, DIM_X> F = kf::Matrix<DIM_X, DIM_X>::Zero();
  for (size_t i = 0; i < DIM_X; i += 2)
  {
    F(i, i) = 1.0F;
    F(i, i + 1) = T;
    F(i + 1, i) = 0.0F;
    F(i + 1, i + 1) = 1.0F;
  }

  kf::Matrix<DIM_X, DIM_X> Q = kf::Matrix<DIM_X, DIM_X>::Zero();
  const kf::float32_t q11 = Q_DIAG * T + Q_DIAG * (std::pow(T, 3.0F) / 3.0F);
  const kf::float32_t q12 = Q_DIAG * (std::pow(T, 2.0F) / 2.0F);
  const kf::float32_t q22 = Q_DIAG * T;

  for (size_t i = 0; i < DIM_X; i += 2)
  {
    Q(i, i) = q11;
    Q(i, i + 1) = q12;
    Q(i + 1, i) = q12;
    Q(i + 1, i + 1) = q22;
  }

  kalmanfilter.predictLKF(F, Q);

  std::cout << "\npredicted state vector =\n" << kalmanfilter.vecX() << "\n";
  std::cout << "\npredicted state covariance=\n"
            << kalmanfilter.matP().topLeftCorner<4, 4>() << "\n";
}

void executeCorrectionStep()
{
  kf::Vector<DIM_Z> vecZ;
  for (size_t i = 0; i < DIM_Z; ++i)
  {
    vecZ(i) = 2.25F;
  }

  kf::Matrix<DIM_Z, DIM_Z> matR = kf::Matrix<DIM_Z, DIM_Z>::Identity() * 0.01F;

  kf::Matrix<DIM_Z, DIM_X> matH = kf::Matrix<DIM_Z, DIM_X>::Zero();
  for (size_t i = 0; i < DIM_Z; ++i)
  {
    matH(i, 2 * i) = 1.0F;
  }

  kalmanfilter.correctLKF(vecZ, matR, matH);

  std::cout << "\ncorrected state vector =\n" << kalmanfilter.vecX() << "\n";
  std::cout << "\ncorrected state covariance =\n"
            << kalmanfilter.matP().topLeftCorner<4, 4>() << "\n";
}