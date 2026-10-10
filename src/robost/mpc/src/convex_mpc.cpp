#include "robost_mpc/convex_mpc.hpp"

#include <chrono>
#include <vector>

#include <osqp.h>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

namespace robost_mpc {

Mat3 RotZ(double yaw) {
  const double c = std::cos(yaw), s = std::sin(yaw);
  Mat3 R;
  R << c, -s, 0,
       s,  c, 0,
       0,  0, 1;
  return R;
}

Mat3 Skew(const Vec3& v) {
  Mat3 S;
  S <<     0, -v.z(),  v.y(),
       v.z(),      0, -v.x(),
      -v.y(),  v.x(),      0;
  return S;
}

Mat3 MpcParams::BoxInertia(double mass, double length, double width, double height) {
  const double k = mass / 12.0;
  return Eigen::Vector3d(k * (width * width + height * height),
                         k * (length * length + height * height),
                         k * (length * length + width * width))
      .asDiagonal();
}

// ---------------------------------------------------------------------------
// 1. System model (paper Eq. 16):
//
//   d/dt [Theta]   [0 0 Rz^T 0] [Theta]   [   0          ...     0        ]       [0]
//        [p    ] = [0 0  0   I] [p    ] + [   0          ...     0        ] u  +  [0]
//        [omega]   [0 0  0   0] [omega]   [I^-1[r1]x     ... I^-1[r4]x    ]       [0]
//        [p_dot]   [0 0  0   0] [p_dot]   [  I/m         ...    I/m       ]       [g]
//
// Small roll/pitch: Theta_dot ~= Rz(yaw)^T omega and I_world ~= Rz I_body Rz^T.
// ---------------------------------------------------------------------------
Mat12 ConvexMpc::ContinuousA(double yaw) {
  Mat12 A = Mat12::Zero();
  A.block<3, 3>(0, 6) = RotZ(yaw).transpose();
  A.block<3, 3>(3, 9) = Mat3::Identity();
  return A;
}

Mat12 ConvexMpc::ContinuousB(double yaw, const FootArray& r_foot) const {
  const Mat3 Rz = RotZ(yaw);
  const Mat3 I_world_inv = (Rz * params_.inertia_body * Rz.transpose()).inverse();
  Mat12 B = Mat12::Zero();
  for (int leg = 0; leg < kNumLegs; ++leg) {
    B.block<3, 3>(6, 3 * leg) = I_world_inv * Skew(r_foot[leg]);
    B.block<3, 3>(9, 3 * leg) = Mat3::Identity() / params_.mass;
  }
  return B;
}

// ---------------------------------------------------------------------------
// 2a. Zero-order hold: exp([[Ac Bc gc]; [0 0 0]] * dt) = [[Ad Bd gd]; [0 I 0 ...]]
// ---------------------------------------------------------------------------
void ConvexMpc::Discretize(const Mat12& Ac, const Mat12& Bc, Mat12* Ad, Mat12* Bd,
                           Vec12* gd) const {
  constexpr int n = kStateDim + kInputDim + 1;
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
  M.block(0, 0, kStateDim, kStateDim) = Ac;
  M.block(0, kStateDim, kStateDim, kInputDim) = Bc;
  M(11, n - 1) = -params_.gravity;  // p_ddot_z += -g
  const Eigen::MatrixXd E = (M * params_.dt).exp();
  *Ad = E.block(0, 0, kStateDim, kStateDim);
  *Bd = E.block(0, kStateDim, kStateDim, kInputDim);
  *gd = E.block(0, n - 1, kStateDim, 1);
}

namespace {

using SpMat = Eigen::SparseMatrix<OSQPFloat, Eigen::ColMajor, OSQPInt>;

// OSQP keeps raw pointers into the Eigen storage; `m` must outlive the solver.
OSQPCscMatrix ToOsqp(SpMat& m) {
  m.makeCompressed();
  OSQPCscMatrix out;
  OSQPCscMatrix_set_data(&out, m.rows(), m.cols(), m.nonZeros(), m.valuePtr(),
                         m.innerIndexPtr(), m.outerIndexPtr());
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// 2b. Condensed QP over U = [u0; ...; u_{k-1}]:
//
//   X = A_qp x0 + B_qp U + C_qp                     (C_qp: accumulated gravity)
//   J = ||X - X_ref||_L^2 + ||U||_K^2
//     = 1/2 U^T H U + U^T q + const,
//   H = 2 (B_qp^T L B_qp + K),  q = 2 B_qp^T L (A_qp x0 + C_qp - X_ref)
//
// 2c. Per leg and step (c = 1 stance / 0 swing):
//   -inf <= fx - mu fz <= 0,   0 <= fx + mu fz <= inf
//   -inf <= fy - mu fz <= 0,   0 <= fy + mu fz <= inf
//   c f_min <= fz <= c f_max   (swing => f = 0 through the cone)
// ---------------------------------------------------------------------------
MpcSolution ConvexMpc::Solve(const MpcProblem& problem) const {
  const auto t_start = std::chrono::steady_clock::now();
  const int k = params_.horizon;
  const int nx = kStateDim, nu = kInputDim;
  const int n_var = nu * k;
  MpcSolution sol;

  // Linearize about the current yaw. A and g do not depend on the feet; B is
  // rebuilt per step when per-step lever arms are given (feet that land
  // inside the horizon), otherwise the current ones are used throughout.
  const double yaw = problem.x0(2);
  const Mat12 Ac = ContinuousA(yaw);
  Mat12 Ad;
  Vec12 gd;
  std::vector<Mat12> Bd_steps(k);
  for (int j = 0; j < k; ++j) {
    const bool per_step = j < static_cast<int>(problem.r_foot_steps.size());
    if (j > 0 && !per_step) {
      Bd_steps[j] = Bd_steps[0];
      continue;
    }
    const FootArray& r_f = per_step ? problem.r_foot_steps[j] : problem.r_foot;
    Discretize(Ac, ContinuousB(yaw, r_f), &Ad, &Bd_steps[j], &gd);
  }

  // Prediction matrices.
  std::vector<Mat12> A_pow(k + 1);
  A_pow[0] = Mat12::Identity();
  for (int i = 1; i <= k; ++i) A_pow[i] = Ad * A_pow[i - 1];

  Eigen::MatrixXd A_qp(nx * k, nx);
  Eigen::MatrixXd B_qp = Eigen::MatrixXd::Zero(nx * k, n_var);
  Eigen::VectorXd C_qp(nx * k);
  Vec12 c = Vec12::Zero();
  for (int i = 0; i < k; ++i) {
    A_qp.block(i * nx, 0, nx, nx) = A_pow[i + 1];
    c = Ad * c + gd;
    C_qp.segment(i * nx, nx) = c;
    for (int j = 0; j <= i; ++j) {
      B_qp.block(i * nx, j * nu, nx, nu) = A_pow[i - j] * Bd_steps[j];
    }
  }

  // Cost.
  Eigen::VectorXd L_diag(nx * k);
  Eigen::VectorXd X_ref(nx * k);
  for (int i = 0; i < k; ++i) {
    L_diag.segment(i * nx, nx) = params_.state_weights;
    X_ref.segment(i * nx, nx) = problem.x_ref.col(i);
  }
  const Eigen::MatrixXd BtL = B_qp.transpose() * L_diag.asDiagonal();
  Eigen::MatrixXd H = 2.0 * BtL * B_qp;
  H.diagonal().array() += 2.0 * params_.input_weight;
  Eigen::VectorXd q = 2.0 * BtL * (A_qp * problem.x0 + C_qp - X_ref);

  // OSQP takes the upper triangle of H.
  std::vector<Eigen::Triplet<OSQPFloat, OSQPInt>> trip;
  trip.reserve(n_var * (n_var + 1) / 2);
  for (int col = 0; col < n_var; ++col)
    for (int row = 0; row <= col; ++row)
      if (H(row, col) != 0.0) trip.emplace_back(row, col, H(row, col));
  SpMat P(n_var, n_var);
  P.setFromTriplets(trip.begin(), trip.end());

  // Friction cone + normal force bounds.
  const int n_cone = 5 * kNumLegs * k;
  const bool torque_rows = !problem.torque_bound.empty();
  const int n_con = n_cone + (torque_rows ? 3 * kNumLegs * k : 0);
  const double mu = params_.mu;
  Eigen::VectorXd lo(n_con), hi(n_con);
  trip.clear();
  for (int i = 0; i < k; ++i) {
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const int col = i * nu + 3 * leg;   // fx, fy, fz at col, col+1, col+2
      const int row = (i * kNumLegs + leg) * 5;
      const double stance = problem.contact(leg, i) ? 1.0 : 0.0;
      trip.emplace_back(row + 0, col + 0, 1.0); trip.emplace_back(row + 0, col + 2, -mu);
      trip.emplace_back(row + 1, col + 0, 1.0); trip.emplace_back(row + 1, col + 2, mu);
      trip.emplace_back(row + 2, col + 1, 1.0); trip.emplace_back(row + 2, col + 2, -mu);
      trip.emplace_back(row + 3, col + 1, 1.0); trip.emplace_back(row + 3, col + 2, mu);
      trip.emplace_back(row + 4, col + 2, 1.0);
      const double f_max = problem.fz_max.cols() > i ? problem.fz_max(leg, i) : params_.f_max;
      lo.segment<5>(row) << -OSQP_INFTY, 0.0, -OSQP_INFTY, 0.0, stance * params_.f_min;
      hi.segment<5>(row) << 0.0, OSQP_INFTY, 0.0, OSQP_INFTY, stance * f_max;
    }
  }
  if (torque_rows) {
    for (int i = 0; i < k; ++i)
      for (int leg = 0; leg < kNumLegs; ++leg) {
        const JointTorqueBound& b = problem.torque_bound[i][leg];
        const int col = i * nu + 3 * leg;
        for (int r = 0; r < 3; ++r) {
          const int row = n_cone + (i * kNumLegs + leg) * 3 + r;
          for (int c = 0; c < 3; ++c)
            if (b.G(r, c) != 0.0) trip.emplace_back(row, col + c, b.G(r, c));
          lo(row) = b.lo(r);
          hi(row) = b.hi(r);
        }
      }
  }
  SpMat A_con(n_con, n_var);
  A_con.setFromTriplets(trip.begin(), trip.end());

  // Solve. A fresh setup per call keeps the example simple; for speed keep the
  // solver alive and use osqp_update_data_mat / osqp_update_data_vec instead.
  OSQPCscMatrix P_csc = ToOsqp(P);
  OSQPCscMatrix A_csc = ToOsqp(A_con);
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.eps_abs = 1e-5;
  settings.eps_rel = 1e-5;
  settings.max_iter = 4000;
  settings.polishing = 0;

  OSQPSolver* solver = nullptr;
  if (osqp_setup(&solver, &P_csc, q.data(), &A_csc, lo.data(), hi.data(), n_con, n_var,
                 &settings) == 0) {
    osqp_solve(solver);
    sol.status = static_cast<int>(solver->info->status_val);
    sol.iterations = static_cast<int>(solver->info->iter);
    sol.ok = sol.status == OSQP_SOLVED || sol.status == OSQP_SOLVED_INACCURATE;
    if (sol.ok) sol.forces = Eigen::Map<const Vec12>(solver->solution->x);
  }
  osqp_cleanup(solver);

  sol.solve_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start)
          .count();
  return sol;
}

}  // namespace robost_mpc
