#include "robost_mpc/actuator_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace robost_mpc {
namespace {

constexpr double kRpm = 2.0 * M_PI / 60.0;

std::vector<std::pair<double, double>> FromRpm(std::vector<std::pair<double, double>> pts) {
  for (auto& p : pts) p.first *= kRpm;
  return pts;
}

// Piecewise-linear interpolation, clamped at both ends. xs ascending.
double Interp(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  const size_t i = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
  const double t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

}  // namespace

// RS02 manual 1.3 / 1.4: peak 17 N m, rated 6 N m, 410 rpm no load, 7.75:1.
// T-N points read off the 48 V curve; output shaft speed.
MotorSpec MotorSpec::RS02() {
  return {"RS02", 17.0, 6.0, 0.0042,
          FromRpm({{220, 17.0}, {250, 15.3}, {300, 12.0}, {350, 7.7}, {380, 4.0}, {400, 1.5},
                   {410, 0.0}})};
}

// RS06 manual 1.3 / 1.4: peak 36 N m, rated 11 N m, 9:1. The 48 V curve ends at
// about 430 rpm (below the 480 rpm quoted no-load speed), so no torque is
// assumed beyond it.
MotorSpec MotorSpec::RS06() {
  return {"RS06", 36.0, 11.0, 0.012,
          FromRpm({{280, 36.0}, {300, 33.0}, {350, 26.0}, {390, 20.0}, {415, 15.0}, {428, 10.0},
                   {430, 5.0}, {432, 0.0}})};
}

double MotorSpec::MaxTorque(double speed) const {
  const double s = std::abs(speed);
  if (s <= tn_curve.front().first) return peak_torque;
  if (s >= tn_curve.back().first) return 0.0;
  size_t i = 1;
  while (tn_curve[i].first < s) ++i;
  const auto& a = tn_curve[i - 1];
  const auto& b = tn_curve[i];
  return a.second + (s - a.first) / (b.first - a.first) * (b.second - a.second);
}

ActuatorModel::ActuatorModel(mjModel* m, const std::array<int, 12>& dof,
                             const std::array<int, 12>& act, const std::string& knee_csv,
                             JointCaps caps, double strength_scale)
    : m_(m), dof_(dof), act_(act), strength_(strength_scale) {
  for (int j = 0; j < 12; ++j) qpos_[j] = m->jnt_qposadr[m->dof_jntid[dof[j]]];
  joint_cap_ = caps == JointCaps::kUrdf ? std::array<double, 3>{17.0, 23.0, 30.0}
                                        : std::array<double, 3>{rs02_.peak_torque,
                                                                rs06_.peak_torque, 30.0};

  std::ifstream in(knee_csv);
  if (!in) throw std::runtime_error("cannot open knee linkage table " + knee_csv);
  std::string line;
  std::getline(in, line);  // header: phi_deg,q_calf_rad,motor_rel_rad,ratio,...
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> v;
    while (std::getline(ss, cell, ',')) v.push_back(std::stod(cell));
    if (v.size() < 4) continue;
    knee_q_.push_back(v[1]);
    knee_ratio_.push_back(v[3]);
  }
  if (knee_q_.size() < 2) throw std::runtime_error("empty knee linkage table " + knee_csv);
  if (knee_q_.front() > knee_q_.back()) {
    std::reverse(knee_q_.begin(), knee_q_.end());
    std::reverse(knee_ratio_.begin(), knee_ratio_.end());
  }

  for (int j = 0; j < 12; ++j) {
    const double cap = joint_cap_[j % 3] * strength_;
    limit_(j) = cap;
    m_->actuator_ctrlrange[2 * act_[j]] = -cap;
    m_->actuator_ctrlrange[2 * act_[j] + 1] = cap;
    m_->actuator_forcerange[2 * act_[j]] = -cap;
    m_->actuator_forcerange[2 * act_[j] + 1] = cap;
    const int jnt = m_->dof_jntid[dof_[j]];
    m_->jnt_actfrcrange[2 * jnt] = -cap;
    m_->jnt_actfrcrange[2 * jnt + 1] = cap;
  }
}

Vec3 ActuatorModel::StaticLegLimit(const Vec3& q) const {
  const double r = std::abs(KneeRatio(q(2)));
  return Vec3(std::min(joint_cap_[0] * strength_, rs02_.peak_torque * strength_),
              std::min(joint_cap_[1] * strength_, rs06_.peak_torque * strength_),
              std::min(joint_cap_[2] * strength_, r * rs06_.peak_torque * strength_));
}

double ActuatorModel::KneeRatio(double q) const { return Interp(knee_q_, knee_ratio_, q); }

double ActuatorModel::KneeRatioDerivative(double q) const {
  constexpr double h = 1e-3;
  return (KneeRatio(q + h) - KneeRatio(q - h)) / (2.0 * h);
}

void ActuatorModel::PreStep(mjData* d) {
  for (int leg = 0; leg < kNumLegs; ++leg) {
    const int j = 3 * leg + 2;
    const double q = d->qpos[qpos_[j]], v = d->qvel[dof_[j]];
    const double r = KneeRatio(q), dr = KneeRatioDerivative(q);
    const double rotor = rs06_.rotor_inertia;
    m_->dof_armature[dof_[j]] = rotor * r * r;
    // Lagrange: I(q) qddot + 0.5 I'(q) qdot^2 = tau, so the speed term is a force.
    d->qfrc_applied[dof_[j]] = -0.5 * (2.0 * rotor * r * dr) * v * v;
  }
}

Vec12 ActuatorModel::Apply(mjData* d, const Vec12& tau_cmd) {
  Vec12 tau;
  for (int j = 0; j < 12; ++j) {
    const int type = j % 3;
    const double q = d->qpos[qpos_[j]], v = d->qvel[dof_[j]];
    const double r = type == 2 ? std::abs(KneeRatio(q)) : 1.0;
    const MotorSpec& motor = this->motor(j);
    const double motor_speed = r * std::abs(v);
    const bool motoring = tau_cmd(j) * v > 0.0;
    // Motor torque is limited by the T-N curve while it drives the load; while
    // braking it is only current limited, i.e. the peak torque.
    const double motor_max = (motoring ? motor.MaxTorque(motor_speed) : motor.peak_torque) * strength_;
    const double lim = std::min(joint_cap_[type] * strength_, r * motor_max);
    limit_(j) = lim;
    const double cmd = std::isfinite(tau_cmd(j)) ? tau_cmd(j) : 0.0;
    tau(j) = std::clamp(cmd, -lim, lim);
    d->ctrl[act_[j]] = tau(j);

    ActuatorStats& st = stats_[j];
    const double tm = std::abs(tau(j)) / r;
    st.torque_sq_sum += tm * tm;
    st.torque_peak = std::max(st.torque_peak, tm);
    st.speed_peak = std::max(st.speed_peak, motor_speed);
    st.limited_ticks += std::abs(cmd) > lim + 1e-9;
    ++st.ticks;
  }
  return tau;
}

}  // namespace robost_mpc
