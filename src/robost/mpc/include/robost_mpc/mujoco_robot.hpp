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
  // Normal force on each foot [N]. Read() fills it from the simulator's contact data; with a
  // contact estimator the controller gets the estimated value instead (foot_force_true keeps
  // the simulator value for comparison only).
  std::array<double, kNumLegs> foot_force{};
  std::array<double, kNumLegs> foot_force_true{};
  std::array<Mat3, kNumLegs> J_leg;   // d foot_world / d q_leg  (world frame)
  std::array<Vec3, kNumLegs> leg_bias;  // gravity + Coriolis torques of leg joints
  
  // Full dynamics terms for WBIC (nv = 18)
  Eigen::MatrixXd A;            // Mass matrix (18x18)
  Eigen::VectorXd b;            // Coriolis + gravity (18x1)
  std::array<Eigen::MatrixXd, kNumLegs> J_foot; // Full Jacobians for each foot (3x18)
};

class MujocoRobot {
 public:
  MujocoRobot(const mjModel* m, mjData* d);

  // Valid after mj_step1 / mj_forward.
  RobotState Read() const;

  // Leg kinematics at another foot position: inverse kinematics of the three leg joints
  // (base fixed at its current pose) to put the foot at `foot_world`. Gives the 3x3 foot
  // Jacobian and the joint angles there. Returns false if the foot is out of reach.
  bool LegAt(int leg, const Vec3& foot_world, Mat3* J, Vec3* q) const;

  // Clips to the actuator ctrlrange and writes d->ctrl. Superseded in the
  // simulation by ActuatorModel::Apply, which adds the motor T-N envelope.
  void ApplyTorques(const Vec12& tau);

  ~MujocoRobot();
  MujocoRobot(const MujocoRobot&) = delete;
  MujocoRobot& operator=(const MujocoRobot&) = delete;

  int base_body() const { return base_body_; }
  // DOF / actuator index of joint `part` (hip, thigh, calf) of `leg`.
  int dof(int leg, int part) const { return dof_[leg][part]; }
  int actuator(int leg, int part) const { return act_[leg][part]; }

 private:
  const mjModel* m_;
  mjData* d_;
  mjData* scratch_ = nullptr;  // IK workspace
  int base_body_ = -1;
  int root_qpos_ = -1, root_dof_ = -1;
  std::array<int, kNumLegs> site_{};
  std::array<int, kNumLegs> foot_body_{};
  std::array<std::array<int, 3>, kNumLegs> dof_{};
  std::array<std::array<int, 3>, kNumLegs> act_{};
};

}  // namespace robost_mpc
