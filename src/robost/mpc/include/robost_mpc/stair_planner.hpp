// Everything the controller does differently on stairs, kept apart from the
// flat-ground locomotion code. The controller owns a StairPlanner only when the
// terrain is not flat; on flat ground none of this code runs.
//
//   - footholds: snapped onto the tread and set to its height
//   - swing: a leg that crosses a riser goes up, forward, then down
//   - body reference: height and pitch follow the terrain under the feet
//   - crawl reference: CoM over the centroid of the feet that stay down
//   - force limits: normal-force bound per leg from the joint torque budget
//   - gait clock: waits for real touchdown and pulls forward on early contact
#pragma once

#include <array>
#include <functional>
#include <vector>

#include "robost_mpc/convex_mpc.hpp"
#include "robost_mpc/mujoco_robot.hpp"
#include "robost_mpc/terrain.hpp"

namespace robost_mpc {

struct Gait;

struct StairConfig {
  double clearance = 0.06;   // [m] foot clearance above the highest terrain on the swing path
  // A swing that crosses a riser or a drop is lifted straight up first, moves
  // forward at apex height, then descends (fractions of the swing time).
  double rise_end = 0.40;
  double xy_start = 0.30, xy_end = 0.90;
  double fall_start = 0.75;
  // Normal-force bound from the thigh torque: fz <= torque / (|foot - hip| + arm). The
  // MPC does not know joint limits, and a foot far behind or ahead of its hip makes the
  // same force cost much more thigh torque.
  double thigh_torque_budget = 23.0;   // [N m], the thigh cap
  double thigh_force_arm = 0.03;       // [m] allowance for the horizontal force component
  double min_fz_max = 80.0;            // [N]
  double tread_margin = 0.07;          // [m] keep footholds this far from every riser
  double pitch_gain = 1.0;             // fraction of the stair slope the body pitch follows
  double pitch_limit = 0.45;           // [rad]
  double height_offset = 0.0;          // [m] added to the body height reference while climbing
  double height_offset_down = 0.0;     // [m] the same while descending (blended by the slope)
  double terrain_smooth = 0.15;        // [m] half width of the height average for the body reference
  // Contact-gated gait clock: it stops until the feet have really touched down.
  bool contact_gating = true;
  // Waiting before a lift for loaded feet and a centered body made the approach to the
  // first riser worse in tests, so it is off unless asked for.
  bool gate_lift = false;
  double contact_force = 8.0;        // [N] normal force that counts as touching
  double shift_tolerance = 0.035;    // [m] CoM-to-support-centroid distance allowed before a lift
  double max_hold_s = 1.5;           // [s] give up waiting after this long
  double hold_descent_speed = 0.06;  // [m/s] how fast a waiting foot keeps reaching down
  double hold_max_reach = 0.10;      // [m] total extra reach below the planned landing
  // Joint torque limits inside the MPC: for a foot at `foot_world` the callback gives the
  // foot Jacobian and the torque each joint can hold. Without it (or with torque_aware
  // false) the MPC does not know the limits.
  // The body stops moving forward while a front foot on the ground is closer than this ahead
  // of the CoM (0 = off), so the body does not run over feet that wait behind a riser.
  double standoff_min = 0.0;         // [m]
  double standoff_ramp = 0.05;       // [m] speed fades out over this distance
  // A foot that should be on the ground but hangs above its tread is pulled down.
  double seek_kp = 0.0;              // [N/m] (0 = off)
  double seek_max = 60.0;            // [N]
  bool torque_aware = false;
  double torque_margin = 0.9;        // fraction of the limit the MPC may use
  std::function<bool(int leg, const Vec3& foot_world, Mat3* J, Vec3* limit)> leg_model;
};

class StairPlanner {
 public:
  // `terrain` must outlive the planner. `ground_foot_z` is the foot site height on flat ground.
  StairPlanner(const Terrain* terrain, const StairConfig& cfg, double ground_foot_z,
               double physics_dt, int ticks_per_mpc, const FootArray& initial_feet);

  static bool Needed(const Terrain* terrain) { return terrain != nullptr && !terrain->flat(); }
  const StairConfig& config() const { return cfg_; }

  // --- footholds ---
  // Snap the xy target onto its tread and put it at the tread height.
  Vec3 PlaceFoot(Vec3 target) const;
  // Touchdown target for a swing leg: follows `fresh` until mid-swing, then is frozen so a
  // snap to another tread cannot jump the foot.
  Vec3 Landing(int leg, double swing_progress, const Vec3& fresh);

  // --- swing ---
  // Swing target at progress `p` toward the touchdown `pf`, aimed `touchdown_drop` below the
  // tread (plus the extra reach while waiting for contact).
  void Swing(int leg, const Vec3& p0, Vec3 pf, double touchdown_drop, double p, double t_swing,
             Vec3* pos, Vec3* vel) const;

  // --- MPC reference ---
  // Body height and pitch under the predicted footprint at time `t` ahead.
  void BodyReference(const RobotState& s, const Vec3& v_des, double t, double z_des,
                     const FootArray& nominal_foot, double* z, double* pitch) const;
  // With one to three feet down, put the CoM xy reference (prob->x_ref) over their centroid.
  void CrawlReference(const std::vector<FootArray>& foot_plan, MpcProblem* prob) const;
  // Per-leg normal-force bounds (prob->fz_max); needs the final x_ref.
  void ForceLimits(const std::vector<FootArray>& foot_plan, const RobotState& s,
                   const FootArray& nominal_foot, double f_max, MpcProblem* prob) const;

  // Joint torque bounds on the ground reaction forces (prob->torque_bound).
  void TorqueBounds(const std::vector<FootArray>& foot_plan, const RobotState& s,
                    MpcProblem* prob) const;

  // Scale the forward command (v_des.x) down when a front foot on the ground is too close
  // ahead of the CoM.
  void LimitVelocity(const RobotState& s, const std::array<bool, kNumLegs>& stance,
                     Vec3* v_des) const;

  // Extra force on a scheduled-stance foot that is not touching its tread (world frame, on the foot).
  Vec3 ContactSeeking(int leg, const RobotState& s) const;

  // --- gait clock ---
  // Call once per tick before computing the gait position. Returns true if the clock waits
  // this tick (it then shifts the schedule by one tick). May pull the clock forward.
  bool Gate(const Gait& gait, double segment_pos, const RobotState& s, const FootArray& liftoff);
  // After the stance/swing split of this tick is final.
  void AfterTick(const std::array<bool, kNumLegs>& stance);
  long clock_offset() const { return hold_total_; }  // ticks the schedule is shifted by
  bool holding() const { return holding_; }
  double reach_extra(int leg) const { return reach_extra_[leg]; }
  int early_touchdowns() const { return early_count_; }

 private:
  const Terrain* terrain_;
  StairConfig cfg_;
  double ground_foot_z_, dt_;
  int ticks_per_mpc_;
  FootArray landing_;
  long hold_total_ = 0;   // gait clock ticks spent waiting (shifts the schedule)
  long event_hold_ = 0;   // ticks waited at the current event
  int early_count_ = 0;
  bool holding_ = false;
  std::array<double, kNumLegs> reach_extra_{};  // extra downward reach while waiting for touchdown
};

}  // namespace robost_mpc
