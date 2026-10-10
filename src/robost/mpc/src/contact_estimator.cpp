#include "robost_mpc/contact_estimator.hpp"

#include <algorithm>
#include <cmath>

namespace robost_mpc {

ContactEstimator::ContactEstimator(const std::array<int, 12>& dof, const Eigen::VectorXd& damping,
                                   double dt, const SensorNoise& noise, double cutoff_hz)
    : dof_(dof), damping_(damping), dt_(dt), noise_(noise), rng_(noise.seed) {
  const double rc = 1.0 / (2.0 * M_PI * cutoff_hz);
  alpha_ = dt / (dt + rc);
}

std::array<double, kNumLegs> ContactEstimator::Update(const RobotState& s,
                                                      const Eigen::VectorXd& qvel,
                                                      const Eigen::VectorXd& base_qacc,
                                                      const Eigen::VectorXd& known_applied,
                                                      const Vec12& tau_applied) {
  const int nv = static_cast<int>(qvel.size());
  const double k = noise_.scale;
  Eigen::VectorXd v_meas = qvel;
  for (int i = 6; i < nv; ++i) v_meas(i) += k * noise_.joint_vel * gauss_(rng_);

  Eigen::VectorXd acc = Eigen::VectorXd::Zero(nv);
  if (have_prev_) acc.tail(nv - 6) = (v_meas.tail(nv - 6) - v_prev_.tail(nv - 6)) / dt_;
  for (int i = 0; i < 3; ++i) acc(i) = base_qacc(i) + k * noise_.imu_lin_acc * gauss_(rng_);
  for (int i = 3; i < 6; ++i) acc(i) = base_qacc(i) + k * noise_.imu_ang_acc * gauss_(rng_);
  v_prev_ = v_meas;
  const bool first = !have_prev_;
  have_prev_ = true;

  // Joint torque residual of every leg joint.
  const Eigen::VectorXd Aacc = s.A * acc;
  for (int j = 0; j < 12; ++j) {
    const int i = dof_[j];
    const double tau_meas = tau_applied(j) + k * noise_.torque * gauss_(rng_);
    const double r = Aacc(i) + s.b(i) + damping_(i) * v_meas(i) - known_applied(i) - tau_meas;
    resid_(j) = first ? r : resid_(j) + alpha_ * (r - resid_(j));
  }

  std::array<double, kNumLegs> fz{};
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const Mat3& J = s.J_leg[leg];
    const Vec3 tau_ext = resid_.segment<3>(3 * leg);
    // J^T F = tau_ext, damped least squares.
    const Vec3 F = (J * J.transpose() + 1e-4 * Mat3::Identity()).ldlt().solve(J * tau_ext);
    fz[leg] = std::max(0.0, F.z());
  }
  return fz;
}

}  // namespace robost_mpc
