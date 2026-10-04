// Gait schedule + stance (MPC) / swing (Cartesian PD) leg control.
#pragma once

#include <array>
#include <cmath>

#include "robost_mpc/convex_mpc.hpp"
#include "robost_mpc/mujoco_robot.hpp"

namespace robost_mpc {

// Offset/duration gait over `segments` MPC steps (MIT Cheetah gait format).
struct Gait {
  const char* name;
  int segments;
  std::array<int, kNumLegs> offsets;
  std::array<int, kNumLegs> durations;  // stance length in segments

  // Position of `leg` inside its own cycle, in segments [0, segments).
  double LegPhase(int leg, double segment_pos) const {
    double s = std::fmod(segment_pos - offsets[leg], segments);
    return s < 0 ? s + segments : s;
  }
  bool Stance(int leg, double segment_pos) const {
    return LegPhase(leg, segment_pos) < durations[leg];
  }
  // 0..1 progress through the current swing.
  double SwingProgress(int leg, double segment_pos) const {
    return (LegPhase(leg, segment_pos) - durations[leg]) / (segments - durations[leg]);
  }

  static Gait Stand() { return {"stand", 10, {0, 0, 0, 0}, {10, 10, 10, 10}}; }
  // FR+RL and FL+RR move together.
  static Gait Trot() { return {"trot", 10, {0, 5, 5, 0}, {5, 5, 5, 5}}; }
};

struct Command {
  double vx = 0.0, vy = 0.0;  // body-frame forward / lateral [m/s]
  double yaw_rate = 0.0;      // [rad/s]
};

struct ControllerConfig {
  MpcParams mpc;
  int ticks_per_mpc = 33;       // 33 x 1 ms physics steps = 33 ms (30 Hz)
  double physics_dt = 0.001;
  double swing_height = 0.06;   // [m]
  double foothold_kv = 0.03;    // Raibert velocity feedback gain [s]
  Vec3 swing_kp = Vec3(500, 500, 500);  // [N/m]
  Vec3 swing_kd = Vec3(12, 12, 12);     // [N s/m]
};

class LocomotionController {
 public:
  // `initial` defines the nominal stance (foot layout, CoM height, ground).
  LocomotionController(const ControllerConfig& cfg, const RobotState& initial);

  // Takes effect at the next MPC tick so gait segments stay aligned with MPC.
  void SetGait(const Gait& gait, long tick);
  void SetCommand(const Command& cmd) { cmd_ = cmd; }

  // Call once per physics tick; runs the MPC when tick % ticks_per_mpc == 0.
  Vec12 Update(long tick, const RobotState& s);

  const MpcSolution& last_solution() const { return last_; }
  const Gait& gait() const { return gait_; }
  const Vec12& forces() const { return forces_; }

 private:
  void RunMpc(double segment_pos, const RobotState& s);
  Vec3 Foothold(int leg, const RobotState& s) const;

  ControllerConfig cfg_;
  ConvexMpc mpc_;
  Gait gait_ = Gait::Stand();
  long gait_start_tick_ = 0;
  Command cmd_;

  FootArray nominal_foot_;  // foot - CoM in the yaw frame (xy only)
  double z_des_;            // CoM height reference
  Eigen::Vector2d xy_des_;  // CoM xy reference integrated from command
  double ground_foot_z_;    // foot site height when touching the ground
  double yaw_des_;

  Vec12 forces_ = Vec12::Zero();  // held (ZOH) between MPC solves
  std::array<bool, kNumLegs> was_stance_{true, true, true, true};
  FootArray liftoff_;
  MpcSolution last_;
};

}  // namespace robost_mpc
