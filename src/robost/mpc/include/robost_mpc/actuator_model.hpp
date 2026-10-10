// RobStride actuator model for the RS06 quadruped joints.
//
//   hip (ab/ad)  : RobStride 02, direct drive at the joint
//   thigh        : RobStride 06, direct drive at the joint
//   calf (knee)  : RobStride 06 through a four-bar linkage (knee_linkage.csv)
//
// Motor data are from the RobStride 02 / 06 user manuals (48 V): the T-N
// curve bounds output torque while the motor drives the load (tau * w > 0);
// braking is current limited, so the full peak torque is available there.
// The knee ratio r(q) = d(motor angle)/d(knee angle) maps motor speed
// |r| * |qdot| and joint torque |r| * tau_motor, and gives the reflected rotor
// inertia I_rotor * r(q)^2 that changes with knee angle.
#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

#include "robost_mpc/convex_mpc.hpp"

namespace robost_mpc {

struct MotorSpec {
  const char* name;
  double peak_torque;     // [N m] at the motor output shaft
  double rated_torque;    // [N m] continuous
  double rotor_inertia;   // [kg m^2] reflected to the output shaft
  // Output-shaft T-N curve, (speed [rad/s], max torque [N m]), speed ascending.
  std::vector<std::pair<double, double>> tn_curve;

  double MaxTorque(double speed) const;  // motoring envelope at |speed|

  static MotorSpec RS02();
  static MotorSpec RS06();
};

enum class JointCaps {
  kUrdf,   // project caps from actuator_params.yaml: 17 / 23 / 30 N m
  kMotor,  // motor peaks only, plus the 30 N m knee structural cap
};

// Per-joint statistics over a run, motor side.
struct ActuatorStats {
  double torque_sq_sum = 0.0;  // for RMS
  double torque_peak = 0.0;
  double speed_peak = 0.0;     // [rad/s] motor output shaft
  long limited_ticks = 0;      // commanded torque was cut by the envelope
  long ticks = 0;
};

class ActuatorModel {
 public:
  // `knee_csv`: assets/rs06/knee_linkage.csv. `dof`/`act` index the 12 joints
  // in controller order (FR, FL, RR, RL x hip, thigh, calf). Also sets the
  // model's actuator and joint torque ranges to the joint caps.
  // `strength_scale` multiplies the whole envelope (1 = the real motors).
  ActuatorModel(mjModel* m, const std::array<int, 12>& dof, const std::array<int, 12>& act,
                const std::string& knee_csv, JointCaps caps, double strength_scale);

  // Before mj_step1: update the knee's reflected inertia for the current angle
  // and apply the matching velocity-dependent term 0.5 I'(q) qdot^2.
  void PreStep(mjData* d);

  // Clip the commanded joint torques to what the motors can deliver at the
  // current joint state and write d->ctrl. Returns the applied torques.
  Vec12 Apply(mjData* d, const Vec12& tau_cmd);

  // Torque each joint of one leg can hold at standstill (no speed derating), for the leg
  // angles `q`. The knee limit depends on the linkage ratio at its angle.
  Vec3 StaticLegLimit(const Vec3& q) const;

  double KneeRatio(double q) const;  // signed d(motor)/d(knee), from the CSV
  double KneeRatioDerivative(double q) const;
  const Vec12& limit() const { return limit_; }  // joint torque envelope at the last Apply
  const std::array<ActuatorStats, 12>& stats() const { return stats_; }
  const MotorSpec& motor(int joint) const { return joint % 3 == 0 ? rs02_ : rs06_; }

 private:
  mjModel* m_;
  std::array<int, 12> dof_, qpos_, act_;
  MotorSpec rs02_ = MotorSpec::RS02(), rs06_ = MotorSpec::RS06();
  std::array<double, 3> joint_cap_{};
  double strength_;
  std::vector<double> knee_q_, knee_ratio_;  // ascending q
  Vec12 limit_ = Vec12::Zero();
  std::array<ActuatorStats, 12> stats_{};
};

}  // namespace robost_mpc
