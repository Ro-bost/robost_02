#include "robost_mpc/locomotion_controller.hpp"

#include <algorithm>
#include <cmath>

namespace robost_mpc {

LocomotionController::LocomotionController(const ControllerConfig& cfg,
                                           const RobotState& initial)
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
}

void LocomotionController::SetGait(const Gait& gait, long tick) {
  const long n = cfg_.ticks_per_mpc;
  gait_ = gait;
  gait_start_tick_ = ((tick + n - 1) / n) * n;  // next MPC boundary
}

Vec12 LocomotionController::Update(long tick, const RobotState& s) {
  const long rel = std::max(0L, tick - gait_start_tick_);
  const double segment_pos =
      std::fmod(static_cast<double>(rel) / cfg_.ticks_per_mpc, gait_.segments);

  std::array<bool, kNumLegs> stance;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    stance[leg] = gait_.Stance(leg, segment_pos);
    if (was_stance_[leg] && !stance[leg]) liftoff_[leg] = s.foot_world[leg];
    was_stance_[leg] = stance[leg];
  }

  // 3. MPC at 30 Hz; forces are held constant until the next solve.
  if (tick % cfg_.ticks_per_mpc == 0) RunMpc(segment_pos, s);

  const double t_swing =
      (gait_.segments - gait_.durations[0]) * cfg_.mpc.dt;  // same for all legs here
  Vec12 tau;
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const Mat3& J = s.J_leg[leg];
    Vec3 tau_leg;
    if (stance[leg]) {
      // tau = J_body^T R^T (-f) = -J_world^T f : the leg pushes the ground with
      // -f so the ground pushes the body with the MPC force f. The leg-link
      // gravity/Coriolis term is added because RS06 legs are not massless
      // (without it the thighs droop and the body drifts while standing).
      tau_leg = -J.transpose() * forces_.segment<3>(3 * leg) + s.leg_bias[leg];
    } else {
      // Swing: smooth xy interpolation + sine lift toward the Raibert foothold.
      const double p = std::clamp(gait_.SwingProgress(leg, segment_pos), 0.0, 1.0);
      const Vec3 p0 = liftoff_[leg];
      const Vec3 pf = Foothold(leg, s);
      const double b = p * p * (3.0 - 2.0 * p);
      const double db = 6.0 * p * (1.0 - p) / t_swing;
      Vec3 pos = p0 + (pf - p0) * b;
      Vec3 vel = (pf - p0) * db;
      pos.z() += cfg_.swing_height * std::sin(M_PI * p);
      vel.z() += cfg_.swing_height * M_PI * std::cos(M_PI * p) / t_swing;
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

  MpcProblem prob;
  prob.x0 << s.rpy, s.com, s.omega_world, s.vel_world;

  // Keep the yaw reference on the same branch as the measured yaw.
  yaw_des_ += cmd_.yaw_rate * p.dt;
  yaw_des_ = s.rpy.z() + std::remainder(yaw_des_ - s.rpy.z(), 2.0 * M_PI);

  const Vec3 v_des = RotZ(s.rpy.z()) * Vec3(cmd_.vx, cmd_.vy, 0.0);
  xy_des_ += v_des.head<2>() * p.dt;

  prob.x_ref.resize(kStateDim, k);
  prob.contact.resize(kNumLegs, k);
  for (int n = 0; n < k; ++n) {
    const double t = (n + 1) * p.dt;
    prob.x_ref.col(n) << 0.0, 0.0, yaw_des_ + cmd_.yaw_rate * t,
        xy_des_.x() + v_des.x() * t, xy_des_.y() + v_des.y() * t, z_des_,
        0.0, 0.0, cmd_.yaw_rate,
        v_des.x(), v_des.y(), 0.0;
    for (int leg = 0; leg < kNumLegs; ++leg)
      prob.contact(leg, n) = gait_.Stance(leg, std::floor(segment_pos) + n) ? 1 : 0;
  }
  for (int leg = 0; leg < kNumLegs; ++leg) prob.r_foot[leg] = s.foot_world[leg] - s.com;

  last_ = mpc_.Solve(prob);
  if (last_.ok) forces_ = last_.forces;  // otherwise keep the previous forces
}

Vec3 LocomotionController::Foothold(int leg, const RobotState& s) const {
  const double t_stance = gait_.durations[leg] * cfg_.mpc.dt;
  const Vec3 v_des = RotZ(s.rpy.z()) * Vec3(cmd_.vx, cmd_.vy, 0.0);
  Vec3 shift = s.vel_world * (0.5 * t_stance) + cfg_.foothold_kv * (s.vel_world - v_des);
  shift.z() = 0.0;
  if (shift.norm() > 0.15) shift *= 0.15 / shift.norm();
  Vec3 target = s.com + RotZ(s.rpy.z()) * nominal_foot_[leg] + shift;
  target.z() = ground_foot_z_;  // flat ground assumption
  return target;
}

}  // namespace robost_mpc
