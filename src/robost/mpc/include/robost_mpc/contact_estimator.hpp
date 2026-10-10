// Foot contact force estimated from what a real robot has: joint encoders, the motor
// torque it commanded, an IMU and the dynamics model. No simulator contact data.
//
// Per leg the dynamics give  M qdd + bias + damping * qd = tau + J^T F,  so the external
// torque on the leg joints is  tau_ext = M qdd + bias + damping * qd - tau,  and the ground
// reaction on the foot follows from  J^T F = tau_ext. The leg accelerations come from the
// differentiated encoder speed and the base acceleration from the IMU, both noisy; the
// residual is low-pass filtered.
//
// The noise levels are assumptions, not values from the motor manuals.
#pragma once

#include <array>
#include <random>

#include "robost_mpc/mujoco_robot.hpp"

namespace robost_mpc {

struct SensorNoise {
  double joint_vel = 0.05;     // [rad/s] encoder speed noise (1 sigma)
  double imu_lin_acc = 0.15;   // [m/s^2] base linear acceleration
  double imu_ang_acc = 1.5;    // [rad/s^2] base angular acceleration (differentiated gyro)
  double torque = 0.25;        // [N m] measured motor torque
  double scale = 1.0;          // multiplies all of the above
  unsigned seed = 1;
};

class ContactEstimator {
 public:
  // `dof`: DOF index of the 12 leg joints in controller order. `damping`: model joint damping.
  ContactEstimator(const std::array<int, 12>& dof, const Eigen::VectorXd& damping, double dt,
                   const SensorNoise& noise, double cutoff_hz = 40.0);

  // Call once per tick after the state is read. `qvel`/`base_qacc` are simulator values that
  // stand in for the encoder and IMU (noise is added here), `tau_applied` is the torque the
  // motors delivered during the last step, `known_applied` the model's own joint force term.
  // Returns the estimated normal force on each foot [N] (>= 0).
  std::array<double, kNumLegs> Update(const RobotState& s, const Eigen::VectorXd& qvel,
                                      const Eigen::VectorXd& base_qacc,
                                      const Eigen::VectorXd& known_applied,
                                      const Vec12& tau_applied);

 private:
  std::array<int, 12> dof_;
  Eigen::VectorXd damping_;
  double dt_, alpha_;
  SensorNoise noise_;
  std::mt19937 rng_;
  std::normal_distribution<double> gauss_{0.0, 1.0};
  Eigen::VectorXd v_prev_;
  bool have_prev_ = false;
  Vec12 resid_ = Vec12::Zero();
};

}  // namespace robost_mpc
