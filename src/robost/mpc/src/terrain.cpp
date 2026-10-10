#include "robost_mpc/terrain.hpp"

#include <algorithm>
#include <cmath>

namespace robost_mpc {

Terrain::Terrain(const mjModel* m, const mjData* d, double x_min, double x_max, double dx)
    : x_min_(x_min), dx_(dx) {
  const int n = static_cast<int>(std::ceil((x_max - x_min) / dx)) + 1;
  h_.assign(n, 0.0);
  const mjtByte group[mjNGROUP] = {1, 0, 0, 0, 0, 0};  // terrain only
  constexpr double kTop = 10.0;
  const mjtNum dir[3] = {0.0, 0.0, -1.0};
  for (int i = 0; i < n; ++i) {
    const mjtNum from[3] = {x_min + i * dx, 0.0, kTop};
    int geom = -1;
    const mjtNum dist = mj_ray(m, d, from, dir, group, 1, -1, &geom, nullptr);
    h_[i] = dist >= 0.0 ? kTop - dist : 0.0;
    top_ = std::max(top_, h_[i]);
  }
  for (int i = 0; i + 1 < n; ++i)
    if (std::abs(h_[i + 1] - h_[i]) > kEdgeJump) edges_.push_back(x_min + (i + 0.5) * dx);
}

double Terrain::Height(double x) const {
  const double u = (x - x_min_) / dx_;
  if (u <= 0.0) return h_.front();
  if (u >= h_.size() - 1) return h_.back();
  const int i = static_cast<int>(u);
  const double t = u - i;
  return h_[i] + t * (h_[i + 1] - h_[i]);
}

double Terrain::HeightSmoothed(double x, double half_width) const {
  if (half_width <= 0.0) return Height(x);
  constexpr int kSamples = 9;
  double sum = 0.0;
  for (int i = 0; i < kSamples; ++i)
    sum += Height(x - half_width + 2.0 * half_width * i / (kSamples - 1));
  return sum / kSamples;
}

double Terrain::MaxHeight(double x0, double x1) const {
  if (x0 > x1) std::swap(x0, x1);
  double best = Height(x0);
  for (double x = x0; x <= x1; x += dx_) best = std::max(best, Height(x));
  return std::max(best, Height(x1));
}

double Terrain::SnapToTread(double x, double margin) const {
  if (edges_.empty()) return x;
  // Tread containing x: between the closest edge behind and ahead.
  const auto ahead = std::upper_bound(edges_.begin(), edges_.end(), x);
  const double lo_edge = ahead == edges_.begin() ? -1e9 : *(ahead - 1);
  const double hi_edge = ahead == edges_.end() ? 1e9 : *ahead;
  const double lo = lo_edge + margin, hi = hi_edge - margin;
  if (lo > hi) return 0.5 * (lo_edge + hi_edge);
  return std::clamp(x, lo, hi);
}

}  // namespace robost_mpc
