// Thin C ABI bridge to the unmodified MIT Cheetah FootSwingTrajectory.
#include "Controllers/FootSwingTrajectory.h"
extern "C" void cheetah_swing(const double *start, const double *end,
                              double height, double phase, double duration,
                              double *out) {
  FootSwingTrajectory<double> swing;
  swing.setInitialPosition(Vec3<double>(start[0], start[1], start[2]));
  swing.setFinalPosition(Vec3<double>(end[0], end[1], end[2]));
  swing.setHeight(height);
  swing.computeSwingTrajectoryBezier(phase, duration);
  for (int i=0; i<3; ++i) {
    out[i] = swing.getPosition()[i];
    out[i+3] = swing.getVelocity()[i];
    out[i+6] = swing.getAcceleration()[i];
  }
}
