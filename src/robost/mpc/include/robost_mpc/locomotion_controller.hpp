// Gait schedule + stance (MPC) / swing (Cartesian PD) leg control.
#pragma once

#include <array>
#include <cmath>
#include <memory>

#include "robost_mpc/convex_mpc.hpp"
#include "robost_mpc/mujoco_robot.hpp"
#include "robost_mpc/stair_planner.hpp"
#include "robost_mpc/terrain.hpp"
#include "robost_mpc/wbic.hpp"

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
  // Flying trot: diagonal pairs with aerial flight phase (Kim et al. 2019)
  static Gait FlyingTrot() { return {"flying_trot", 10, {0, 5, 5, 0}, {4, 4, 4, 4}}; }
  // Bound: front pair (FR, FL) and rear pair (RR, RL) alternate (Di Carlo 2018, Kim 2019)
  static Gait Bound() { return {"bound", 10, {0, 0, 5, 5}, {5, 5, 5, 5}}; }
  // Pace: right pair (FR, RR) and left pair (FL, RL) alternate (Di Carlo 2018)
  static Gait Pace() { return {"pace", 10, {0, 5, 0, 5}, {5, 5, 5, 5}}; }
  // Pronk: all four legs jump synchronously with flight phase (Di Carlo 2018, Kim 2019)
  static Gait Pronk() { return {"pronk", 10, {0, 0, 0, 0}, {4, 4, 4, 4}}; }
  // Walk: 4-beat crawl with 3 legs in stance (75% duty factor)
  static Gait Walk() { return {"walk", 12, {0, 6, 9, 3}, {9, 9, 9, 9}}; }
  // Gallop: rotary gallop with staggered sequence (Di Carlo 2018, Kim 2019)
  static Gait Gallop() { return {"gallop", 10, {0, 2, 7, 5}, {4, 4, 4, 4}}; }
  // Stairs: slower versions with longer swings so a foot has time to clear a riser.
  static Gait StairTrot() { return {"stair_trot", 20, {0, 10, 10, 0}, {10, 10, 10, 10}}; }
  // 4-beat crawl, 75% duty: exactly one leg in swing at any time.
  // 80% duty crawl with a 0.1 s all-feet-down interval between swings, which gives the
  // body time to shift over the next support triangle. Cycle 2 s: 0.16 m/s = one 0.32 m tread.
  static Gait StairCrawl() { return {"stair_crawl", 60, {0, 30, 45, 15}, {48, 48, 48, 48}}; }
  // One cycle is 1.6 s and each leg swings for 0.4 s; at 0.2 m/s a foot advances one tread per cycle.
  static Gait StairWalk() { return {"stair_walk", 48, {0, 24, 36, 12}, {36, 36, 36, 36}}; }
};

struct Command {
  double vx = 0.0, vy = 0.0;  // body-frame forward / lateral [m/s]
  double yaw_rate = 0.0;      // [rad/s]
};

struct ControllerConfig {
  MpcParams mpc;
  int ticks_per_mpc = 33;       // MPC prediction step: 33 x 1 ms = 33 ms
  int ticks_per_solve = 10;     // MPC re-solve interval: 10 ms (100 Hz)
  double physics_dt = 0.001;
  double swing_height = 0.05;   // [m] apex above the higher of liftoff / touchdown
  double touchdown_depth = 0.01;  // [m] swing ends this far below ground for firm contact
  double foothold_kv = 0.15;    // [s] velocity feedback (closer to sqrt(h/g) ~ 0.19)
  double max_lin_acc = 1.5;     // [m/s^2] command ramp
  double max_yaw_acc = 3.0;     // [rad/s^2]
  StairConfig stair;            // only used when the terrain is not flat
  Vec3 swing_kp = Vec3(1500, 1500, 1500);  // [N/m]
  Vec3 swing_kd = Vec3(40, 40, 40);     // [N s/m]
};

class LocomotionController {
 public:
  // `initial` defines the nominal stance (foot layout, CoM height, ground).
  // `terrain` may be null (flat ground); it must outlive the controller.
  LocomotionController(const ControllerConfig& cfg, const RobotState& initial,
                       const Terrain* terrain = nullptr);

  // Takes effect at the next MPC tick so gait segments stay aligned with MPC.
  void SetGait(const Gait& gait, long tick);
  void SetCommand(const Command& cmd) { target_cmd_ = cmd; }

  // Call once per physics tick; runs the MPC when tick % ticks_per_solve == 0.
  Vec12 Update(long tick, const RobotState& s);

  const MpcSolution& last_solution() const { return last_; }
  const Gait& gait() const { return gait_; }
  const Vec12& forces() const { return forces_; }
  // Stairs only: gait clock waiting time [ticks] and early touchdowns (0 on flat ground).
  long hold_ticks() const { return stairs_ ? stairs_->clock_offset() : 0; }
  int early_touchdowns() const { return stairs_ ? stairs_->early_touchdowns() : 0; }
  // Scheduled stance per leg from the last Update() (debug/logging).
  const std::array<bool, kNumLegs>& stance() const { return was_stance_; }

 private:
  void RunMpc(double segment_pos, const RobotState& s);
  Vec3 Foothold(int leg, const RobotState& s, double t_ahead = 0.0) const;

  ControllerConfig cfg_;
  std::unique_ptr<StairPlanner> stairs_;  // null on flat ground
  ConvexMpc mpc_;
  Wbic wbic_{WbicParams()};
  Gait gait_ = Gait::Stand();
  long gait_start_tick_ = 0;
  Command target_cmd_;
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
