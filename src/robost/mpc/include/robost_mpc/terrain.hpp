// Terrain height profile along the world x axis, read from the MuJoCo model.
//
// The stair course is uniform across y, so one profile h(x) sampled at y = 0
// is enough. Terrain geoms are the ones in geom group 0 (the robot's geoms are
// in groups 1 and 3), so ray casting straight down hits only the terrain. A
// flat scene gives h(x) = 0 everywhere, which makes every terrain-aware
// computation in the controller reduce to the flat-ground behavior.
#pragma once

#include <vector>

#include <mujoco/mujoco.h>

namespace robost_mpc {

class Terrain {
 public:
  // Needs `d` after mj_forward so that static geom poses are valid.
  Terrain(const mjModel* m, const mjData* d, double x_min = -1.0, double x_max = 14.0,
          double dx = 0.005);

  bool flat() const { return edges_.empty(); }
  double Height(double x) const;                         // linear interpolation
  double HeightSmoothed(double x, double half_width) const;  // box average
  double MaxHeight(double x0, double x1) const;          // over [min, max]
  double TopHeight() const { return top_; }

  // Vertical jumps (risers / drops) larger than `kEdgeJump`, as x positions.
  const std::vector<double>& edges() const { return edges_; }

  // Move x along its tread so that it stays at least `margin` from every
  // riser or drop. Treads shorter than 2 * margin map to their middle.
  double SnapToTread(double x, double margin) const;

 private:
  static constexpr double kEdgeJump = 0.02;
  double x_min_, dx_;
  double top_ = 0.0;
  std::vector<double> h_;
  std::vector<double> edges_;
};

}  // namespace robost_mpc
