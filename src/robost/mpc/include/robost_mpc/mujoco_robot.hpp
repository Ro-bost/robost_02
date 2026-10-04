// Thin MuJoCo adapter for the RS06 scene exported by export_scene.py.
//
// Leg order everywhere: FR, FL, RR, RL (same as robost.simulation.hardware).
#pragma once

#include <array>

#include <mujoco/mujoco.h>

#include "robost_mpc/convex_mpc.hpp"

namespace robost_mpc {

struct RobotState {
  Vec3 rpy;            // roll, pitch, yaw (ZYX)
  Mat3 R;              // body -> world
  Vec3 com;            // whole-robot CoM, world
  Vec3 omega_world;    // base angular velocity, world
  Vec3 vel_world;      // base linear velocity, world
  FootArray foot_world;
  FootArray foot_vel_world;
  std::array<Mat3, kNumLegs> J_leg;   // d foot_world / d q_leg  (world frame)
  std::array<Vec3, kNumLegs> leg_bias;  // gravity + Coriolis torques of leg joints
};

class MujocoRobot {
 public:
  MujocoRobot(const mjModel* m, mjData* d);

  // Valid after mj_step1 / mj_forward.
  RobotState Read() const;

  // Clips to the actuator ctrlrange (URDF effort limits) and writes d->ctrl.
  void ApplyTorques(const Vec12& tau);

  int base_body() const { return base_body_; }

 private:
  const mjModel* m_;
  mjData* d_;
  int base_body_ = -1;
  int root_qpos_ = -1, root_dof_ = -1;
  std::array<int, kNumLegs> site_{};
  std::array<std::array<int, 3>, kNumLegs> dof_{};
  std::array<std::array<int, 3>, kNumLegs> act_{};
};

}  // namespace robost_mpc
