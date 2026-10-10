#pragma once

#include "robost_mpc/mujoco_robot.hpp"
#include <Eigen/Dense>
#include <array>

namespace robost_mpc {

struct WbicParams {
  double friction = 0.6;
  double f_max = 300.0;
  
  double w_f = 10.0;         // Weight for tracking MPC forces
  double w_base = 1e-4;      // Weight for regularizing base acceleration
  double w_swing = 5.0;      // Weight for tracking swing foot acceleration
  double w_reg = 0.1;        // Weight for regularizing joint accelerations
};

class Wbic {
 public:
  explicit Wbic(const WbicParams& params) : params_(params) {}

  // Computes the full body joint torques (size 12)
  // `mpc_forces`: 12D vector of contact forces desired by MPC
  // `s`: Current RobotState containing A, b, J_foot, vel_world, etc.
  // `stance`: Array indicating which legs are in stance
  // `foot_acc_cmd`: Desired foot accelerations (world frame) for each leg.
  Vec12 ComputeTorques(const Vec12& mpc_forces, const RobotState& s,
                       const std::array<bool, kNumLegs>& stance,
                       const std::array<Vec3, kNumLegs>& foot_acc_cmd) const;

 private:
  WbicParams params_;
};

}  // namespace robost_mpc
