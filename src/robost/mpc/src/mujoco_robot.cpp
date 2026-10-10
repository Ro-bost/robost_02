#include "robost_mpc/mujoco_robot.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace robost_mpc {
namespace {

constexpr const char* kLegNames[kNumLegs] = {"FR", "FL", "RR", "RL"};
constexpr const char* kPartNames[3] = {"hip", "thigh", "calf"};

int RequireId(const mjModel* m, mjtObj type, const std::string& name) {
  const int id = mj_name2id(m, type, name.c_str());
  if (id < 0) throw std::runtime_error("MuJoCo model has no object named " + name);
  return id;
}

}  // namespace

MujocoRobot::MujocoRobot(const mjModel* m, mjData* d) : m_(m), d_(d) {
  base_body_ = RequireId(m, mjOBJ_BODY, "base");
  const int root = RequireId(m, mjOBJ_JOINT, "root");
  root_qpos_ = m->jnt_qposadr[root];
  root_dof_ = m->jnt_dofadr[root];
  for (int leg = 0; leg < kNumLegs; ++leg) {
    site_[leg] = RequireId(m, mjOBJ_SITE, kLegNames[leg]);
    foot_body_[leg] = m->site_bodyid[site_[leg]];
    for (int j = 0; j < 3; ++j) {
      const std::string joint = std::string(kLegNames[leg]) + "_" + kPartNames[j] + "_joint";
      dof_[leg][j] = m->jnt_dofadr[RequireId(m, mjOBJ_JOINT, joint)];
      act_[leg][j] = RequireId(m, mjOBJ_ACTUATOR, joint + "_motor");
    }
  }
}

MujocoRobot::~MujocoRobot() {
  if (scratch_) mj_deleteData(scratch_);
}

bool MujocoRobot::LegAt(int leg, const Vec3& foot_world, Mat3* J, Vec3* q_out) const {
  if (!scratch_) const_cast<MujocoRobot*>(this)->scratch_ = mj_makeData(m_);
  mjData* d = scratch_;
  mju_copy(d->qpos, d_->qpos, m_->nq);
  int qadr[3], jid[3];
  for (int j = 0; j < 3; ++j) {
    jid[j] = m_->dof_jntid[dof_[leg][j]];
    qadr[j] = m_->jnt_qposadr[jid[j]];
  }
  std::vector<mjtNum> jacp(3 * m_->nv);
  Mat3 J3 = Mat3::Identity();
  double err_norm = 0.0;
  for (int it = 0; it < 30; ++it) {
    mj_kinematics(m_, d);
    mj_comPos(m_, d);
    const Vec3 err = foot_world - Eigen::Map<const Vec3>(d->site_xpos + 3 * site_[leg]);
    err_norm = err.norm();
    mj_jacSite(m_, d, jacp.data(), nullptr, site_[leg]);
    Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> Jf(jacp.data(), 3,
                                                                                    m_->nv);
    for (int j = 0; j < 3; ++j) J3.col(j) = Jf.col(dof_[leg][j]);
    if (err_norm < 5e-4) break;
    const Vec3 dq = (J3.transpose() * J3 + 1e-4 * Mat3::Identity()).ldlt().solve(J3.transpose() * err);
    for (int j = 0; j < 3; ++j) {
      double q = d->qpos[qadr[j]] + std::clamp(dq(j), -0.2, 0.2);
      if (m_->jnt_limited[jid[j]])
        q = std::clamp(q, m_->jnt_range[2 * jid[j]], m_->jnt_range[2 * jid[j] + 1]);
      d->qpos[qadr[j]] = q;
    }
  }
  *J = J3;
  for (int j = 0; j < 3; ++j) (*q_out)(j) = d->qpos[qadr[j]];
  return err_norm < 5e-3;
}

RobotState MujocoRobot::Read() const {
  RobotState s;
  const mjtNum* quat = d_->qpos + root_qpos_ + 3;  // w, x, y, z
  mjtNum mat[9];
  mju_quat2Mat(mat, quat);
  s.R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(mat);
  s.rpy << std::atan2(s.R(2, 1), s.R(2, 2)),
           std::asin(std::clamp(-s.R(2, 0), -1.0, 1.0)),
           std::atan2(s.R(1, 0), s.R(0, 0));

  s.com = Eigen::Map<const Vec3>(d_->subtree_com + 3 * base_body_);
  // Free joint: linear velocity in world frame, angular velocity in body frame.
  s.vel_world = Eigen::Map<const Vec3>(d_->qvel + root_dof_);
  s.omega_world = s.R * Eigen::Map<const Vec3>(d_->qvel + root_dof_ + 3);

  std::vector<mjtNum> jacp(3 * m_->nv);
  for (int leg = 0; leg < kNumLegs; ++leg) {
    s.foot_world[leg] = Eigen::Map<const Vec3>(d_->site_xpos + 3 * site_[leg]);
    mj_jacSite(m_, d_, jacp.data(), nullptr, site_[leg]);
    Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>> J(jacp.data(), 3,
                                                                                  m_->nv);
    s.foot_vel_world[leg] = J * Eigen::Map<const Eigen::VectorXd>(d_->qvel, m_->nv);
    for (int j = 0; j < 3; ++j) {
      s.J_leg[leg].col(j) = J.col(dof_[leg][j]);
      s.leg_bias[leg](j) = d_->qfrc_bias[dof_[leg][j]];
    }
    s.J_foot[leg] = J;
  }

  // Normal force of the foot body against any static geom (terrain).
  for (int i = 0; i < d_->ncon; ++i) {
    const mjContact& c = d_->contact[i];
    const int b1 = m_->geom_bodyid[c.geom1], b2 = m_->geom_bodyid[c.geom2];
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const bool hit = (b1 == foot_body_[leg] && m_->body_weldid[b2] == 0) ||
                       (b2 == foot_body_[leg] && m_->body_weldid[b1] == 0);
      if (!hit) continue;
      mjtNum f[6];
      mj_contactForce(m_, d_, i, f);
      s.foot_force[leg] += f[0];
    }
  }

  s.foot_force_true = s.foot_force;

  // Full dynamics extraction
  s.A.resize(m_->nv, m_->nv);
  mj_fullM(m_, d_, s.A.data());
  s.b = Eigen::Map<const Eigen::VectorXd>(d_->qfrc_bias, m_->nv);

  return s;
}

void MujocoRobot::ApplyTorques(const Vec12& tau) {
  for (int leg = 0; leg < kNumLegs; ++leg) {
    for (int j = 0; j < 3; ++j) {
      const int a = act_[leg][j];
      const double lo = m_->actuator_ctrlrange[2 * a], hi = m_->actuator_ctrlrange[2 * a + 1];
      const double t = tau(3 * leg + j);
      d_->ctrl[a] = std::isfinite(t) ? std::clamp(t, lo, hi) : 0.0;
    }
  }
}

}  // namespace robost_mpc
