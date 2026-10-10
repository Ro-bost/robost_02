#include "robost_mpc/stair_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "robost_mpc/locomotion_controller.hpp"
#include "robost_mpc/swing.hpp"

namespace robost_mpc {

StairPlanner::StairPlanner(const Terrain* terrain, const StairConfig& cfg, double ground_foot_z,
                           double physics_dt, int ticks_per_mpc, const FootArray& initial_feet)
    : terrain_(terrain),
      cfg_(cfg),
      ground_foot_z_(ground_foot_z),
      dt_(physics_dt),
      ticks_per_mpc_(ticks_per_mpc),
      landing_(initial_feet) {}

Vec3 StairPlanner::PlaceFoot(Vec3 target) const {
  target.x() = terrain_->SnapToTread(target.x(), cfg_.tread_margin);
  target.z() = ground_foot_z_ + terrain_->Height(target.x());
  return target;
}

Vec3 StairPlanner::Landing(int leg, double swing_progress, const Vec3& fresh) {
  if (swing_progress < 0.5) landing_[leg] = fresh;
  return landing_[leg];
}

void StairPlanner::Swing(int leg, const Vec3& p0, Vec3 pf, double touchdown_drop, double p,
                         double t_swing, Vec3* pos, Vec3* vel) const {
  pf.z() -= touchdown_drop + reach_extra_[leg];
  // Apex above the highest terrain between liftoff and touchdown.
  const double path_max =
      terrain_->MaxHeight(std::min(p0.x(), pf.x()) - 0.04, std::max(p0.x(), pf.x()) + 0.04) +
      ground_foot_z_;
  const double apex_base = std::max(std::max(p0.z(), pf.z()), path_max);
  const bool crossing = apex_base - std::min(p0.z(), pf.z()) > 0.03;
  const double z_apex = apex_base + cfg_.clearance;
  if (!crossing) {
    LevelSwing(p0, pf, z_apex, p, t_swing, pos, vel);
    return;
  }
  // Over a riser or drop: straight up, forward at apex height, then down.
  double b, db;
  const double x0 = cfg_.xy_start, x1 = cfg_.xy_end;
  Smooth(std::clamp((p - x0) / (x1 - x0), 0.0, 1.0), &b, &db);
  *pos = p0 + (pf - p0) * b;
  *vel = (pf - p0) * (db / ((x1 - x0) * t_swing));
  if (p < cfg_.rise_end) {
    Smooth(p / cfg_.rise_end, &b, &db);
    pos->z() = p0.z() + (z_apex - p0.z()) * b;
    vel->z() = (z_apex - p0.z()) * db / (cfg_.rise_end * t_swing);
  } else if (p < cfg_.fall_start) {
    pos->z() = z_apex;
    vel->z() = 0.0;
  } else {
    const double len = 1.0 - cfg_.fall_start;
    Smooth((p - cfg_.fall_start) / len, &b, &db);
    pos->z() = z_apex + (pf.z() - z_apex) * b;
    vel->z() = (pf.z() - z_apex) * db / (len * t_swing);
  }
  if (p >= x1) vel->head<2>().setZero();
}

void StairPlanner::BodyReference(const RobotState& s, const Vec3& v_des, double t, double z_des,
                                 const FootArray& nominal_foot, double* z, double* pitch) const {
  // The pitch is the slope of the plane through the front and rear feet, and the height is
  // measured along the body normal, so the legs keep their nominal length on a slope.
  const Mat3 Rz = RotZ(s.rpy.z());
  const Vec3 com_t = s.com + v_des * t;
  double h[kNumLegs];
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const Vec3 foot = com_t + Rz * nominal_foot[leg];
    h[leg] = terrain_->HeightSmoothed(foot.x(), cfg_.terrain_smooth);
  }
  const double front = 0.5 * (h[0] + h[1]), rear = 0.5 * (h[2] + h[3]);
  const double lx = 0.5 * (nominal_foot[0].x() + nominal_foot[1].x()) -
                    0.5 * (nominal_foot[2].x() + nominal_foot[3].x());
  *pitch = std::clamp(-cfg_.pitch_gain * std::atan2(front - rear, lx), -cfg_.pitch_limit, cfg_.pitch_limit);
  // Blend between the climbing and the descending offset by the slope under the footprint.
  const double up = std::clamp(0.5 + (front - rear) / 0.2, 0.0, 1.0);
  const double offset = up * cfg_.height_offset + (1.0 - up) * cfg_.height_offset_down;
  *z = (z_des + offset) / std::cos(*pitch) + 0.5 * (front + rear);
}

void StairPlanner::CrawlReference(const std::vector<FootArray>& foot_plan, MpcProblem* prob) const {
  // The set looks `kLead` steps ahead (only feet that stay down), so the body starts
  // shifting before a leg lifts.
  constexpr int kLead = 3;
  const int k = static_cast<int>(foot_plan.size());
  for (int n = 0; n < k; ++n) {
    Eigen::Vector2d sum = Eigen::Vector2d::Zero();
    int count = 0;
    for (int leg = 0; leg < kNumLegs; ++leg) {
      bool down = true;
      for (int m = n; m <= std::min(k - 1, n + kLead); ++m) down = down && prob->contact(leg, m);
      if (down) {
        sum += foot_plan[n][leg].head<2>();
        ++count;
      }
    }
    if (count >= 1 && count <= 3) {
      prob->x_ref(3, n) = sum.x() / count;
      prob->x_ref(4, n) = sum.y() / count;
    }
  }
}

void StairPlanner::ForceLimits(const std::vector<FootArray>& foot_plan, const RobotState& s,
                               const FootArray& nominal_foot, double f_max,
                               MpcProblem* prob) const {
  const Mat3 Rz = RotZ(s.rpy.z());
  const int k = static_cast<int>(foot_plan.size());
  prob->fz_max.resize(kNumLegs, k);
  for (int n = 0; n < k; ++n)
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const Eigen::Vector2d hip = prob->x_ref.block<2, 1>(3, n) + (Rz * nominal_foot[leg]).head<2>();
      const double reach = std::abs(foot_plan[n][leg].x() - hip.x());
      prob->fz_max(leg, n) = std::clamp(cfg_.thigh_torque_budget / (reach + cfg_.thigh_force_arm),
                                        cfg_.min_fz_max, f_max);
    }
}

void StairPlanner::TorqueBounds(const std::vector<FootArray>& foot_plan, const RobotState& s,
                                MpcProblem* prob) const {
  if (!cfg_.torque_aware || !cfg_.leg_model) return;
  const int k = static_cast<int>(foot_plan.size());
  prob->torque_bound.assign(k, {});
  // A foot changes only when it lands inside the horizon, so the leg model is evaluated
  // once per distinct foot position.
  for (int leg = 0; leg < kNumLegs; ++leg) {
    Vec3 cached_foot = Vec3::Constant(1e9);
    Mat3 J = Mat3::Identity();
    Vec3 limit = Vec3::Zero();
    bool valid = false;
    for (int n = 0; n < k; ++n) {
      if (!prob->contact(leg, n)) continue;
      const Vec3& foot = foot_plan[n][leg];
      if ((foot - cached_foot).norm() > 1e-9) {
        valid = cfg_.leg_model(leg, foot, &J, &limit);
        cached_foot = foot;
      }
      if (!valid) continue;
      const bool planted = (foot - s.foot_world[leg]).norm() < 1e-9;
      const Vec3 bias = planted ? s.leg_bias[leg] : Vec3::Zero();
      JointTorqueBound& b = prob->torque_bound[n][leg];
      b.G = -J.transpose();
      b.lo = -cfg_.torque_margin * limit - bias;
      b.hi = cfg_.torque_margin * limit - bias;
    }
  }
}

void StairPlanner::LimitVelocity(const RobotState& s, const std::array<bool, kNumLegs>& stance,
                                 Vec3* v_des) const {
  if (cfg_.standoff_min <= 0.0 || v_des->x() <= 0.0) return;
  double gap = 1e9;
  for (int leg = 0; leg < 2; ++leg)  // front legs
    if (stance[leg]) gap = std::min(gap, s.foot_world[leg].x() - s.com.x());
  if (gap > 1e8) return;
  const double scale = std::clamp((gap - cfg_.standoff_min) / cfg_.standoff_ramp, 0.0, 1.0);
  v_des->x() *= scale;
}

Vec3 StairPlanner::ContactSeeking(int leg, const RobotState& s) const {
  if (cfg_.seek_kp <= 0.0 || s.foot_force[leg] >= cfg_.contact_force) return Vec3::Zero();
  const double tread = ground_foot_z_ + terrain_->Height(s.foot_world[leg].x());
  const double gap = s.foot_world[leg].z() - tread;
  if (gap < 0.01) return Vec3::Zero();
  return Vec3(0.0, 0.0, -std::min(cfg_.seek_max, cfg_.seek_kp * (gap - 0.01)));
}

bool StairPlanner::Gate(const Gait& gait, double segment_pos, const RobotState& s,
                        const FootArray& liftoff) {
  static const bool debug = std::getenv("GATE_DEBUG") != nullptr;
  holding_ = false;
  if (!cfg_.contact_gating) return false;
  const int n = ticks_per_mpc_;
  const double step = 1.0 / n;
  const long max_hold = static_cast<long>(cfg_.max_hold_s / dt_);
  const bool timed_out = event_hold_ >= max_hold;
  auto touching = [&](int leg) {
    if (s.foot_force[leg] < cfg_.contact_force) return false;
    const double tread = ground_foot_z_ + terrain_->Height(s.foot_world[leg].x());
    return std::abs(s.foot_world[leg].z() - tread) < 0.025;
  };

  bool hold = false;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const double ph = gait.LegPhase(leg, segment_pos);
    const bool swing = ph >= gait.durations[leg];
    if (swing) {
      const double p = gait.SwingProgress(leg, segment_pos);
      // Touchdown earlier than scheduled: jump the clock to the end of this swing.
      if (p >= 0.75 && touching(leg)) {
        hold_total_ -= static_cast<long>(std::ceil((gait.segments - ph) * n));
        ++early_count_;
        event_hold_ = 0;
        return false;
      }
      // Reached a higher planned landing without contact: keep reaching down. Going down a
      // step is not waited for (it made the descent worse).
      if (ph + step >= gait.segments && landing_[leg].z() > liftoff[leg].z() - 0.02 &&
          !touching(leg) && !timed_out) {
        hold = true;
        if (debug && event_hold_ % 100 == 0)
          std::fprintf(stderr, "leg %d waits for touchdown, f=%.1f foot=(%.3f %.3f) landing=(%.3f %.3f) tread=%.3f com=%.3f,%.3f\n", leg, s.foot_force[leg], s.foot_world[leg].x(), s.foot_world[leg].z(), landing_[leg].x(), landing_[leg].z(), ground_foot_z_ + terrain_->Height(s.foot_world[leg].x()), s.com.x(), s.com.z());
      }
    } else if (cfg_.gate_lift && ph + step >= gait.durations[leg]) {
      // About to lift this foot: the other feet must be loaded and the body must already be
      // over their support.
      Eigen::Vector2d sum = Eigen::Vector2d::Zero();
      int count = 0;
      bool loaded = true;
      for (int o = 0; o < kNumLegs; ++o) {
        if (o == leg || !gait.Stance(o, segment_pos)) continue;
        loaded = loaded && s.foot_force[o] >= cfg_.contact_force;
        sum += s.foot_world[o].head<2>();
        ++count;
      }
      const bool centered =
          count == 0 || (s.com.head<2>() - sum / count).norm() < cfg_.shift_tolerance;
      if ((!loaded || !centered || s.vel_world.head<2>().norm() > 0.08) && !timed_out) hold = true;
    }
  }
  event_hold_ = hold ? event_hold_ + 1 : 0;
  if (hold) ++hold_total_;
  holding_ = hold;
  return hold;
}

void StairPlanner::AfterTick(const std::array<bool, kNumLegs>& stance) {
  for (int leg = 0; leg < kNumLegs; ++leg) {
    if (stance[leg]) reach_extra_[leg] = 0.0;
    else if (holding_)
      reach_extra_[leg] = std::min(cfg_.hold_max_reach, reach_extra_[leg] + cfg_.hold_descent_speed * dt_);
  }
}

}  // namespace robost_mpc
