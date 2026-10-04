// Convex MPC for quadruped locomotion.
//
// Reference: J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, S. Kim,
// "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive
// Control", IROS 2018.
//
// State   x = [Theta(roll, pitch, yaw), p(com), omega(world), p_dot(world)]  (12)
// Input   u = [f_1, f_2, f_3, f_4]  ground reaction forces in world frame  (12)
//
// Gravity is kept as a constant affine term instead of the paper's extra 13th
// state; both give the same prediction.
#pragma once

#include <array>

#include <Eigen/Dense>

namespace robost_mpc {

constexpr int kNumLegs = 4;
constexpr int kStateDim = 12;
constexpr int kInputDim = 3 * kNumLegs;

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
using FootArray = std::array<Vec3, kNumLegs>;

struct MpcParams {
  int horizon = 10;          // k
  double dt = 0.033;         // MPC step [s] (30 Hz)
  double mass = 18.08;       // lumped body mass [kg]
  Mat3 inertia_body = Mat3::Identity();  // body-frame inertia about the CoM
  double gravity = 9.81;
  double mu = 0.6;           // friction coefficient
  double f_min = 0.0;        // stance-leg normal force bounds [N]
  double f_max = 200.0;
  // Diagonal of L, ordered like x: [roll, pitch, yaw, x, y, z, wx, wy, wz, vx, vy, vz]
  Vec12 state_weights = (Vec12() << 25.0, 25.0, 10.0, 20.0, 20.0, 100.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
                            .finished();
  double input_weight = 1e-4;  // K = alpha * I

  // Solid box inertia: I = m/12 * diag(W^2+H^2, L^2+H^2, L^2+W^2).
  static Mat3 BoxInertia(double mass, double length, double width, double height);
};

struct MpcProblem {
  Vec12 x0;                                    // measured state
  Eigen::Matrix<double, 12, Eigen::Dynamic> x_ref;  // column n = desired x[n+1]
  FootArray r_foot;                            // foot position minus CoM (world)
  Eigen::Matrix<int, kNumLegs, Eigen::Dynamic> contact;  // 1 = stance at step n
};

struct MpcSolution {
  bool ok = false;
  int status = 0;            // OSQP status_val
  int iterations = 0;
  double solve_ms = 0.0;     // matrix build + QP time
  Vec12 forces = Vec12::Zero();  // first input u[0], applied until next solve
};

class ConvexMpc {
 public:
  explicit ConvexMpc(const MpcParams& params) : params_(params) {}

  MpcSolution Solve(const MpcProblem& problem) const;

  // Continuous dynamics x_dot = A_c x + B_c u + g_c (small roll/pitch).
  static Mat12 ContinuousA(double yaw);
  Mat12 ContinuousB(double yaw, const FootArray& r_foot) const;

  // Exact zero-order-hold discretization of [A_c, B_c, g_c].
  void Discretize(const Mat12& Ac, const Mat12& Bc, Mat12* Ad, Mat12* Bd, Vec12* gd) const;

  const MpcParams& params() const { return params_; }

 private:
  MpcParams params_;
};

Mat3 RotZ(double yaw);
Mat3 Skew(const Vec3& v);

}  // namespace robost_mpc
