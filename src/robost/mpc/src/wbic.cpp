#include "robost_mpc/wbic.hpp"

#include <osqp.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <Eigen/Sparse>

namespace robost_mpc {

using SpMat = Eigen::SparseMatrix<OSQPFloat, Eigen::ColMajor, OSQPInt>;

namespace {
OSQPCscMatrix ToOsqp(SpMat& m) {
  m.makeCompressed();
  OSQPCscMatrix out;
  OSQPCscMatrix_set_data(&out, m.rows(), m.cols(), m.nonZeros(), m.valuePtr(),
                         m.innerIndexPtr(), m.outerIndexPtr());
  return out;
}
}  // namespace

Vec12 Wbic::ComputeTorques(const Vec12& mpc_forces, const RobotState& s,
                           const std::array<bool, kNumLegs>& stance,
                           const std::array<Vec3, kNumLegs>& foot_acc_cmd) const {
  const int n_var = 30;  // 18 qddot + 12 fr
  const int n_q = 18;
  const int n_f = 12;

  // Cost P (30x30)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_var, n_var);
  // Base acceleration cost
  H.block(0, 0, 6, 6) = params_.w_base * Eigen::MatrixXd::Identity(6, 6);
  // Joint acceleration cost
  H.block(6, 6, 12, 12) = params_.w_reg * Eigen::MatrixXd::Identity(12, 12);
  
  // Swing foot tracking cost
  for (int leg = 0; leg < kNumLegs; ++leg) {
    if (!stance[leg]) {
      const Eigen::MatrixXd& J = s.J_foot[leg]; // 3x18
      H.block(0, 0, 18, 18) += params_.w_swing * J.transpose() * J;
    }
  }
  
  // Force tracking cost
  H.block(18, 18, 12, 12) = params_.w_f * Eigen::MatrixXd::Identity(12, 12);
  H *= 2.0;  // 1/2 x^T H x

  // Cost q (30x1)
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n_var);
  for (int leg = 0; leg < kNumLegs; ++leg) {
    if (!stance[leg]) {
      q.segment<18>(0) -= 2.0 * params_.w_swing * s.J_foot[leg].transpose() * foot_acc_cmd[leg];
    }
  }
  q.segment<12>(18) = -2.0 * params_.w_f * mpc_forces;

  // OSQP upper triangle
  std::vector<Eigen::Triplet<OSQPFloat, OSQPInt>> trip;
  for (int col = 0; col < n_var; ++col) {
    for (int row = 0; row <= col; ++row) {
      if (H(row, col) != 0.0) trip.emplace_back(row, col, H(row, col));
    }
  }
  SpMat P_mat(n_var, n_var);
  P_mat.setFromTriplets(trip.begin(), trip.end());

  // Constraints A_qp
  // 6 base dynamics + 12 stance constraints + 20 friction cone
  const int n_con = 6 + 12 + 20;
  Eigen::MatrixXd A_qp = Eigen::MatrixXd::Zero(n_con, n_var);
  Eigen::VectorXd lo = Eigen::VectorXd::Constant(n_con, -OSQP_INFTY);
  Eigen::VectorXd hi = Eigen::VectorXd::Constant(n_con, OSQP_INFTY);

  // 1. Base dynamics
  A_qp.block(0, 0, 6, 18) = s.A.topRows(6);
  for (int leg = 0; leg < kNumLegs; ++leg) {
    A_qp.block(0, 18 + 3 * leg, 6, 3) = -s.J_foot[leg].leftCols(6).transpose();
  }
  lo.segment<6>(0) = -s.b.head<6>();
  hi.segment<6>(0) = -s.b.head<6>();

  // 2. Foot constraints
  for (int leg = 0; leg < kNumLegs; ++leg) {
    A_qp.block(6 + 3 * leg, 0, 3, 18) = s.J_foot[leg];
    if (stance[leg]) {
      lo.segment<3>(6 + 3 * leg) = foot_acc_cmd[leg];
      hi.segment<3>(6 + 3 * leg) = foot_acc_cmd[leg];
    }
  }

  // 3. Friction cone (5 inequalities per leg)
  // fx + mu fz >= 0, -fx + mu fz >= 0, fy + mu fz >= 0, -fy + mu fz >= 0, 0 <= fz <= f_max
  for (int leg = 0; leg < kNumLegs; ++leg) {
    int row = 18 + 5 * leg;
    int col = 18 + 3 * leg;
    A_qp(row, col) = 1.0;     A_qp(row, col + 2) = params_.friction; // fx + mu fz
    A_qp(row + 1, col) = -1.0; A_qp(row + 1, col + 2) = params_.friction; // -fx + mu fz
    A_qp(row + 2, col + 1) = 1.0; A_qp(row + 2, col + 2) = params_.friction; // fy + mu fz
    A_qp(row + 3, col + 1) = -1.0; A_qp(row + 3, col + 2) = params_.friction; // -fy + mu fz
    A_qp(row + 4, col + 2) = 1.0; // fz

    if (stance[leg]) {
      for (int i = 0; i < 4; ++i) {
        lo(row + i) = 0.0;
        hi(row + i) = OSQP_INFTY;
      }
      lo(row + 4) = 0.0;
      hi(row + 4) = params_.f_max;
    } else {
      for (int i = 0; i < 5; ++i) {
        lo(row + i) = 0.0;
        hi(row + i) = 0.0;
      }
    }
  }

  trip.clear();
  for (int col = 0; col < n_var; ++col) {
    for (int row = 0; row < n_con; ++row) {
      if (A_qp(row, col) != 0.0) trip.emplace_back(row, col, A_qp(row, col));
    }
  }
  SpMat A_sp(n_con, n_var);
  A_sp.setFromTriplets(trip.begin(), trip.end());

  OSQPCscMatrix P_csc = ToOsqp(P_mat);
  OSQPCscMatrix A_csc = ToOsqp(A_sp);

  OSQPSolver* solver = nullptr;
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.eps_abs = 1e-4;
  settings.eps_rel = 1e-4;
  settings.max_iter = 1000;
  
  if (osqp_setup(&solver, &P_csc, q.data(), &A_csc, lo.data(), hi.data(), n_con, n_var, &settings) == 0) {
    osqp_solve(solver);
  }

  Eigen::VectorXd qddot = Eigen::VectorXd::Zero(n_q);
  Eigen::VectorXd f_res = Eigen::VectorXd::Zero(n_f);
  if (solver && (solver->info->status_val == OSQP_SOLVED || solver->info->status_val == OSQP_SOLVED_INACCURATE)) {
    qddot = Eigen::Map<Eigen::VectorXd>(solver->solution->x, n_q);
    f_res = Eigen::Map<Eigen::VectorXd>(solver->solution->x + n_q, n_f);
  }

  if (solver) osqp_cleanup(solver);

  Eigen::VectorXd tau_full = s.A.bottomRows(12) * qddot + s.b.tail(12);
  for (int leg = 0; leg < kNumLegs; ++leg) {
    tau_full -= s.J_foot[leg].rightCols(12).transpose() * f_res.segment<3>(3 * leg);
  }

  return tau_full;
}

}  // namespace robost_mpc
