#include "robost_mpc/locomotion_controller.hpp"

#include "robost_mpc/swing.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace robost_mpc {
LocomotionController::LocomotionController(const ControllerConfig& cfg, const RobotState& initial,
                                           const Terrain* terrain)
    : cfg_(cfg), mpc_(cfg.mpc) {
  const Mat3 Rz_T = RotZ(initial.rpy.z()).transpose();
  ground_foot_z_ = 0.0;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    nominal_foot_[leg] = Rz_T * (initial.foot_world[leg] - initial.com);
    nominal_foot_[leg].z() = 0.0;
    ground_foot_z_ += initial.foot_world[leg].z() / kNumLegs;
    liftoff_[leg] = initial.foot_world[leg];
  }
  z_des_ = initial.com.z();
  yaw_des_ = initial.rpy.z();
  xy_des_ = initial.com.head<2>();
  if (StairPlanner::Needed(terrain))
    stairs_ = std::make_unique<StairPlanner>(terrain, cfg.stair, ground_foot_z_, cfg.physics_dt,
                                             cfg.ticks_per_mpc, initial.foot_world);
}

void LocomotionController::SetGait(const Gait& gait, long tick) {
  const long n = cfg_.ticks_per_mpc;
  gait_ = gait;
  // Enter the cycle at the middle of leg 0's stance instead of its start. In
  // a periodic gait the body's pitch/roll rate is zero at mid-stance, which
  // matches the robot starting at rest; entering at touchdown instead throws
  // the first half-cycle to one side (bound/pace/gallop lurch on start).
  const long phase0 = (gait.durations[0] + 1) / 2;
  gait_start_tick_ = ((tick + n - 1) / n) * n - phase0 * n;
}

Vec12 LocomotionController::Update(long tick, const RobotState& s) {
  auto position = [&]() {
    const long offset = stairs_ ? stairs_->clock_offset() : 0;
    const long rel = std::max(0L, tick - gait_start_tick_ - offset);
    return std::fmod(static_cast<double>(rel) / cfg_.ticks_per_mpc, gait_.segments);
  };
  double segment_pos = position();
  if (stairs_ && std::strcmp(gait_.name, "stand") != 0) {
    stairs_->Gate(gait_, segment_pos, s, liftoff_);  // may shift the gait clock
    segment_pos = position();
  }

  std::array<bool, kNumLegs> stance;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    stance[leg] = gait_.Stance(leg, segment_pos);
    if (was_stance_[leg] && !stance[leg]) liftoff_[leg] = s.foot_world[leg];
    was_stance_[leg] = stance[leg];
  }
  if (stairs_) stairs_->AfterTick(stance);

  // Ramp the command with bounded acceleration: a step from rest to a fast
  // speed otherwise demands one huge push from the first diagonal pair.
  auto ramp = [](double cur, double target, double max_step) {
    return cur + std::clamp(target - cur, -max_step, max_step);
  };
  const double dt = cfg_.physics_dt;
  cmd_.vx = ramp(cmd_.vx, target_cmd_.vx, cfg_.max_lin_acc * dt);
  cmd_.vy = ramp(cmd_.vy, target_cmd_.vy, cfg_.max_lin_acc * dt);
  cmd_.yaw_rate = ramp(cmd_.yaw_rate, target_cmd_.yaw_rate, cfg_.max_yaw_acc * dt);
  // The MPC predicts with dt = ticks_per_mpc but is re-solved every
  // ticks_per_solve; forces are held constant between solves.
  if (tick % cfg_.ticks_per_solve == 0) RunMpc(segment_pos, s);

  Vec12 tau;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const Mat3& J = s.J_leg[leg];
    Vec3 tau_leg;
    if (stance[leg]) {
      // tau = -J_world^T f: the leg pushes the ground with -f. The leg-link
      // gravity/Coriolis term is added because RS06 legs are ~half the mass.
      tau_leg = -J.transpose() * forces_.segment<3>(3 * leg) + s.leg_bias[leg];
      if (stairs_) tau_leg += J.transpose() * stairs_->ContactSeeking(leg, s);
    } else {
      // Swing: smoothstep in xy; z rises to the apex and comes back down, both
      // halves with zero end velocity so the foot lands softly instead of
      // arriving at the sine's peak downward speed.
      const double t_swing = (gait_.segments - gait_.durations[leg]) * cfg_.mpc.dt;
      const double p = std::clamp(gait_.SwingProgress(leg, segment_pos), 0.0, 1.0);
      const Vec3 p0 = liftoff_[leg];
      Vec3 pf = Foothold(leg, s);
      Vec3 pos, vel;
      if (stairs_) {
        pf = stairs_->Landing(leg, p, pf);
        stairs_->Swing(leg, p0, pf, cfg_.touchdown_depth, p, t_swing, &pos, &vel);
      } else {
        pf.z() -= cfg_.touchdown_depth;
        LevelSwing(p0, pf, std::max(p0.z(), pf.z()) + cfg_.swing_height, p, t_swing, &pos, &vel);
      }

      const Vec3 force = cfg_.swing_kp.cwiseProduct(pos - s.foot_world[leg]) +
                         cfg_.swing_kd.cwiseProduct(vel - s.foot_vel_world[leg]);
      tau_leg = J.transpose() * force + s.leg_bias[leg];
    }
    tau.segment<3>(3 * leg) = tau_leg;
  }
  return tau;
}

void LocomotionController::RunMpc(double segment_pos, const RobotState& s) {
  const MpcParams& p = cfg_.mpc;
  const int k = p.horizon;
  const double solve_dt = cfg_.ticks_per_solve * cfg_.physics_dt;
  const bool is_stand = std::strcmp(gait_.name, "stand") == 0;

  MpcProblem prob;
  prob.x0 << s.rpy, s.com, s.omega_world, s.vel_world;

  // Keep the yaw reference on the same branch as the measured yaw.
  yaw_des_ += cmd_.yaw_rate * solve_dt;
  yaw_des_ = s.rpy.z() + std::remainder(yaw_des_ - s.rpy.z(), 2.0 * M_PI);

  Vec3 v_des = (is_stand || (stairs_ && stairs_->holding())) ? Vec3::Zero() : Vec3(RotZ(s.rpy.z()) * Vec3(cmd_.vx, cmd_.vy, 0.0));
  if (stairs_) {
    std::array<bool, kNumLegs> st;
    for (int leg = 0; leg < kNumLegs; ++leg) st[leg] = gait_.Stance(leg, segment_pos + 1e-6);
    stairs_->LimitVelocity(s, st, &v_des);
  }
  xy_des_ += v_des.head<2>() * solve_dt;

  // Body height and pitch follow the terrain under the predicted footprint.
  // The pitch is the slope of the plane through the front and rear feet, and
  // the height is measured along the body normal, so the legs keep their
  // nominal length on a slope. On flat ground this is (z_des_, 0).
  auto body_ref = [&](double t, double* z, double* pitch) {
    if (!stairs_) {
      *z = z_des_;
      *pitch = 0.0;
      return;
    }
    stairs_->BodyReference(s, v_des, t, z_des_, nominal_foot_, z, pitch);
  };

  prob.x_ref.resize(kStateDim, k);
  prob.contact.resize(kNumLegs, k);
  const double seg0 = segment_pos + 1e-6;
  double z_prev, pitch_unused;
  body_ref(0.0, &z_prev, &pitch_unused);
  for (int n = 0; n < k; ++n) {
    const double t = (n + 1) * p.dt;
    double z_ref, pitch_ref;
    body_ref(t, &z_ref, &pitch_ref);
    prob.x_ref.col(n) << 0.0, pitch_ref, yaw_des_ + cmd_.yaw_rate * t,
        xy_des_.x() + v_des.x() * t, xy_des_.y() + v_des.y() * t, z_ref,
        0.0, 0.0, cmd_.yaw_rate,
        v_des.x(), v_des.y(), (z_ref - z_prev) / p.dt;
    z_prev = z_ref;
    for (int leg = 0; leg < kNumLegs; ++leg)
      prob.contact(leg, n) = gait_.Stance(leg, seg0 + n) ? 1 : 0;
  }

  // Lever arms per horizon step: a foot already on the ground stays where it
  // is; a foot that lands inside the horizon is placed at its planned
  // foothold. Both are taken relative to the predicted CoM at that step.
  for (int leg = 0; leg < kNumLegs; ++leg) prob.r_foot[leg] = s.foot_world[leg] - s.com;
  prob.r_foot_steps.resize(k);
  std::vector<FootArray> foot_plan(k);
  for (int leg = 0; leg < kNumLegs; ++leg) {
    Vec3 foot = s.foot_world[leg];
    bool on_ground = gait_.Stance(leg, seg0);
    for (int n = 0; n < k; ++n) {
      const bool st = prob.contact(leg, n);
      if (st && !on_ground) foot = Foothold(leg, s, n * p.dt);
      on_ground = st;
      foot_plan[n][leg] = foot;
      prob.r_foot_steps[n][leg] = foot - (s.com + v_des * (n * p.dt));
    }
  }

  // Stairs: crawl CoM reference and per-leg force bounds.
  if (stairs_) {
    stairs_->CrawlReference(foot_plan, &prob);
    stairs_->ForceLimits(foot_plan, s, nominal_foot_, p.f_max, &prob);
    stairs_->TorqueBounds(foot_plan, s, &prob);
  }

  last_ = mpc_.Solve(prob);
  if (last_.ok) forces_ = last_.forces;
}

// Raibert heuristic with capture-point feedback. `t_ahead` shifts the CoM
// prediction for footholds of touchdowns later in the MPC horizon.
Vec3 LocomotionController::Foothold(int leg, const RobotState& s, double t_ahead) const {
  const double t_stance = gait_.durations[leg] * cfg_.mpc.dt;
  const Vec3 v_des = RotZ(s.rpy.z()) * Vec3(cmd_.vx, cmd_.vy, 0.0);
  const Vec3 p_centrifugal =
      0.5 * std::sqrt(s.com.z() / cfg_.mpc.gravity) * s.vel_world.cross(Vec3(0, 0, cmd_.yaw_rate));

  Vec3 shift = v_des * (0.5 * t_stance) + cfg_.foothold_kv * (s.vel_world - v_des) + p_centrifugal;
  shift.z() = 0.0;
  shift.x() = std::clamp(shift.x(), -0.16, 0.16);
  shift.y() = std::clamp(shift.y(), -0.18, 0.18);

  Vec3 com = s.com;
  com.head<2>() += v_des.head<2>() * t_ahead;
  Vec3 target = com + RotZ(s.rpy.z()) * nominal_foot_[leg] + shift;
  if (stairs_) return stairs_->PlaceFoot(target);
  target.z() = ground_foot_z_;
  return target;
}

}  // namespace robost_mpc
