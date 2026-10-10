// Swing-foot helpers shared by flat-ground and stair swings.
#pragma once

#include "robost_mpc/convex_mpc.hpp"

namespace robost_mpc {

// Smoothstep with zero velocity at both ends: value and d/ds.
inline void Smooth(double s, double* b, double* db) {
  *b = s * s * (3.0 - 2.0 * s);
  *db = 6.0 * s * (1.0 - s);
}

// Level swing: smoothstep in xy; z rises to `z_apex` and comes back down, both
// halves with zero end velocity so the foot lands softly.
inline void LevelSwing(const Vec3& p0, const Vec3& pf, double z_apex, double p, double t_swing,
                       Vec3* pos, Vec3* vel) {
  double b, db;
  Smooth(p, &b, &db);
  *pos = p0 + (pf - p0) * b;
  *vel = (pf - p0) * (db / t_swing);
  const bool rising = p < 0.5;
  const double z_from = rising ? p0.z() : z_apex, z_to = rising ? z_apex : pf.z();
  Smooth(rising ? 2.0 * p : 2.0 * p - 1.0, &b, &db);
  pos->z() = z_from + (z_to - z_from) * b;
  vel->z() = (z_to - z_from) * db * 2.0 / t_swing;
}

}  // namespace robost_mpc
