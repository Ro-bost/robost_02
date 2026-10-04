// RS06 Convex MPC in MuJoCo: physics at 1000 Hz, MPC at 30 Hz.
//
//   rs06_mpc [--model runs/mpc/scene_flat.mjb] [--gait stand|trot]
//            [--vx 0.2] [--vy 0] [--yaw-rate 0] [--duration 10] [--headless]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <mujoco/mujoco.h>

#include "robost_mpc/locomotion_controller.hpp"
#include "robost_mpc/mujoco_robot.hpp"

#ifdef ROBOST_MPC_WITH_VIEWER
#include <GLFW/glfw3.h>
#endif

using namespace robost_mpc;

namespace {

struct Options {
  std::string model = "runs/mpc/scene_flat.mjb";
  std::string gait = "trot";
  Command cmd{0.2, 0.0, 0.0};
  double duration = 0.0;  // 0: run until the window closes (10 s headless)
  bool headless = false;
};

Options ParseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") o.model = next();
    else if (a == "--gait") o.gait = next();
    else if (a == "--vx") o.cmd.vx = std::atof(next());
    else if (a == "--vy") o.cmd.vy = std::atof(next());
    else if (a == "--yaw-rate") o.cmd.yaw_rate = std::atof(next());
    else if (a == "--duration") o.duration = std::atof(next());
    else if (a == "--headless") o.headless = true;
    else throw std::runtime_error("unknown argument " + a);
  }
  if (o.gait != "stand" && o.gait != "trot") throw std::runtime_error("--gait: stand|trot");
  if (o.headless && o.duration <= 0) o.duration = 10.0;
  return o;
}

mjModel* LoadModel(const std::string& path) {
  char err[1000] = "";
  mjModel* m = nullptr;
  if (path.size() > 4 && path.substr(path.size() - 4) == ".mjb")
    m = mj_loadModel(path.c_str(), nullptr);
  else
    m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (!m) throw std::runtime_error("cannot load " + path + " " + err +
                                   "\n  -> run: python src/robost/mpc/export_scene.py");
  return m;
}

// The whole simulation: one physics tick + controller.
class Simulation {
 public:
  Simulation(mjModel* m, const Options& opt) : m_(m), opt_(opt) {
    m_->opt.timestep = 0.001;  // 1000 Hz physics
    d_ = mj_makeData(m_);
    const int key = mj_name2id(m_, mjOBJ_KEY, "stand");
    if (key >= 0) mj_resetDataKeyframe(m_, d_, key);
    mj_forward(m_, d_);
    robot_ = std::make_unique<MujocoRobot>(m_, d_);

    ControllerConfig cfg;
    cfg.physics_dt = m_->opt.timestep;
    cfg.ticks_per_mpc = static_cast<int>(std::lround(cfg.mpc.dt / cfg.physics_dt));  // 33
    // Box model with the RS06 total mass; dimensions chosen so the box matches
    // the whole-robot inertia at the standing pose (~0.29, 0.87, 0.93 kg m^2).
    double mass = 0.0;
    for (int b = 0; b < m_->nbody; ++b) mass += m_->body_mass[b];
    cfg.mpc.mass = mass;
    cfg.mpc.inertia_body = MpcParams::BoxInertia(mass, 0.70, 0.34, 0.28);
    controller_ = std::make_unique<LocomotionController>(cfg, robot_->Read());
    controller_->SetCommand({0, 0, 0});
  }
  ~Simulation() { mj_deleteData(d_); }

  // Returns false once the robot has fallen.
  bool Step() {
    if (tick_ == kStandTicks) {  // stand still first, then start the gait
      controller_->SetGait(opt_.gait == "trot" ? Gait::Trot() : Gait::Stand(), tick_);
      controller_->SetCommand(opt_.cmd);
    }
    mj_step1(m_, d_);                       // kinematics for the current state
    const RobotState s = robot_->Read();
    robot_->ApplyTorques(controller_->Update(tick_, s));
    mj_step2(m_, d_);                       // integrate with the new d->ctrl
    ++tick_;
    last_ = s;
    return s.com.z() > 0.15 && std::abs(s.rpy.x()) < 1.0 && std::abs(s.rpy.y()) < 1.0;
  }

  void Print() const {
    const MpcSolution& sol = controller_->last_solution();
    const Vec12& f = controller_->forces();
    std::printf("t=%6.2fs gait=%-5s com=(%6.3f %6.3f %5.3f) rpy=(%6.3f %6.3f %6.3f) "
                "vx=%6.3f sum_fz=%6.1fN | QP %s it=%4d %5.2fms\n",
                d_->time, controller_->gait().name, last_.com.x(), last_.com.y(),
                last_.com.z(), last_.rpy.x(), last_.rpy.y(), last_.rpy.z(), last_.vel_world.x(),
                f(2) + f(5) + f(8) + f(11), sol.ok ? "ok " : "ERR", sol.iterations, sol.solve_ms);
    if (std::getenv("MPC_DEBUG")) {
      std::printf("   f=");
      for (int i = 0; i < 12; ++i) std::printf("%6.1f%s", f(i), i % 3 == 2 ? " |" : " ");
      std::printf("\n   ctrl=");
      for (int i = 0; i < 12; ++i) std::printf("%6.1f ", d_->ctrl[i]);
      std::printf("\n");
    }
  }

  mjModel* m() { return m_; }
  mjData* d() { return d_; }
  long tick() const { return tick_; }
  int base_body() const { return robot_->base_body(); }

 private:
  static constexpr long kStandTicks = 500;  // 0.5 s
  mjModel* m_;
  mjData* d_;
  Options opt_;
  std::unique_ptr<MujocoRobot> robot_;
  std::unique_ptr<LocomotionController> controller_;
  RobotState last_;
  long tick_ = 0;
};

int RunHeadless(Simulation& sim, double duration) {
  const auto start = std::chrono::steady_clock::now();
  while (sim.d()->time < duration) {
    if (!sim.Step()) {
      sim.Print();
      std::printf("FALLEN at t=%.3f s\n", sim.d()->time);
      return 1;
    }
    if (sim.tick() % 500 == 0) sim.Print();
  }
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("Finished %.1f s of simulation in %.2f s wall (%.1fx real time)\n", duration, wall,
              duration / wall);
  return 0;
}

#ifdef ROBOST_MPC_WITH_VIEWER
int RunViewer(Simulation& sim, double duration) {
  if (!glfwInit()) throw std::runtime_error("glfwInit failed (no DISPLAY?)");
  GLFWwindow* window = glfwCreateWindow(1280, 900, "RS06 Convex MPC", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjvCamera cam;
  mjvOption vopt;
  mjvScene scn;
  mjrContext con;
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&vopt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);
  mjv_makeScene(sim.m(), &scn, 10000);
  mjr_makeContext(sim.m(), &con, mjFONTSCALE_150);
  cam.type = mjCAMERA_TRACKING;
  cam.trackbodyid = sim.base_body();
  cam.distance = 2.0;
  cam.azimuth = 135;
  cam.elevation = -20;

  bool fallen = false;
  const auto wall_start = std::chrono::steady_clock::now();
  while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
    // Advance physics until simulated time catches up with wall time.
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    while (!fallen && sim.d()->time < wall) {
      if (!sim.Step()) {
        fallen = true;
        sim.Print();
        std::printf("FALLEN at t=%.3f s (close the window to exit)\n", sim.d()->time);
      }
      if (sim.tick() % 500 == 0) sim.Print();
    }
    if (duration > 0 && sim.d()->time >= duration) break;

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(sim.m(), sim.d(), &vopt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    char info[128];
    std::snprintf(info, sizeof(info), "%.2f s%s", sim.d()->time, fallen ? "  FALLEN" : "");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "time", info, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  glfwDestroyWindow(window);
  glfwTerminate();
  return fallen ? 1 : 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = ParseArgs(argc, argv);
    mjModel* m = LoadModel(opt.model);
    int rc = 0;
    {
      Simulation sim(m, opt);
      std::printf("RS06 Convex MPC | gait=%s cmd=(%.2f, %.2f, %.2f) | physics 1000 Hz, MPC 30 Hz\n",
                  opt.gait.c_str(), opt.cmd.vx, opt.cmd.vy, opt.cmd.yaw_rate);
#ifdef ROBOST_MPC_WITH_VIEWER
      rc = opt.headless ? RunHeadless(sim, opt.duration) : RunViewer(sim, opt.duration);
#else
      rc = RunHeadless(sim, opt.duration > 0 ? opt.duration : 10.0);
#endif
    }
    mj_deleteModel(m);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
