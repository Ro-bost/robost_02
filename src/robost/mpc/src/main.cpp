// RS06 Convex MPC in MuJoCo: physics at 1000 Hz, MPC at 30 Hz.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "robost_mpc/actuator_model.hpp"
#include "robost_mpc/contact_estimator.hpp"
#include "robost_mpc/locomotion_controller.hpp"
#include "robost_mpc/mujoco_robot.hpp"
#include "robost_mpc/terrain.hpp"

#ifdef ROBOST_MPC_WITH_VIEWER
#include <GLFW/glfw3.h>
#endif

using namespace robost_mpc;

namespace {

struct Options {
  std::string model = "runs/mpc/scene_flat.mjb";
  std::string gait = "trot";
  Command cmd{0.2, 0.0, 0.0};
  double duration = 0.0;
  bool headless = false;
  double speed = 1.0;
  double torque_scale = 1.0;  // scales the whole motor envelope (1 = the real RS02 / RS06)
  bool motor_caps = false;    // allow the RS06 thigh its 36 N m peak instead of the 23 N m cap
  std::string knee_csv = "assets/rs06/knee_linkage.csv";
  double stair_pitch_weight = 40.0;  // MPC pitch weight when the scene has stairs
  double stair_roll_weight = 100.0;  // MPC roll weight when the scene has stairs
  double tread_margin = 0.07;        // [m] footholds stay this far from every riser
  double stair_clearance = 0.06;     // [m] swing apex above the highest terrain on the path
  bool contact_gating = true;        // stairs: gait clock waits for real contact
  bool sensed_contact = false;       // contact from motor torques + IMU instead of simulator data
  double noise_scale = 1.0;          // sensor noise multiplier for the estimator
  double standoff = 0.16;            // stairs: min front-foot gap ahead of the CoM
  double pitch_gain = 1.0, pitch_limit = 0.45, height_offset = -0.12, height_offset_down = -0.12, seek = 0.0;
  bool torque_aware = false;         // stairs: joint torque limits inside the MPC
  double torque_margin = 0.9;
  bool gate_lift = false;            // also wait for a centered, loaded stance before each lift
  double shift_tol = 0.035;          // [m] CoM-to-support-centroid distance allowed before a lift
  double thigh_budget = 0.0;         // [N m] thigh torque budget for the MPC force bound (0 = off)
  double stair_pos_weight = 300.0;    // MPC x / y position weight on stairs (CoM over the support)
};

Gait ResolveGait(const std::string& name) {
  if (name == "stand") return Gait::Stand();
  if (name == "trot") return Gait::Trot();
  if (name == "flying_trot") return Gait::FlyingTrot();
  if (name == "bound") return Gait::Bound();
  if (name == "pace") return Gait::Pace();
  if (name == "pronk") return Gait::Pronk();
  if (name == "walk") return Gait::Walk();
  if (name == "gallop") return Gait::Gallop();
  if (name == "stair_trot") return Gait::StairTrot();
  if (name == "stair_walk") return Gait::StairWalk();
  if (name == "stair_crawl") return Gait::StairCrawl();
  throw std::runtime_error("unknown gait: " + name);
}

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
    else if (a == "--speed") o.speed = std::atof(next());
    else if (a == "--torque-scale") o.torque_scale = std::atof(next());
    else if (a == "--motor-caps") o.motor_caps = true;
    else if (a == "--knee-csv") o.knee_csv = next();
    else if (a == "--stair-pitch-weight") o.stair_pitch_weight = std::atof(next());
    else if (a == "--stair-roll-weight") o.stair_roll_weight = std::atof(next());
    else if (a == "--stair-pos-weight") o.stair_pos_weight = std::atof(next());
    else if (a == "--thigh-budget") o.thigh_budget = std::atof(next());
    else if (a == "--tread-margin") o.tread_margin = std::atof(next());
    else if (a == "--stair-clearance") o.stair_clearance = std::atof(next());
    else if (a == "--no-gating") o.contact_gating = false;
    else if (a == "--gate-lift") o.gate_lift = true;
    else if (a == "--torque-aware") o.torque_aware = true;
    else if (a == "--standoff") o.standoff = std::atof(next());
    else if (a == "--sensed-contact") o.sensed_contact = true;
    else if (a == "--noise-scale") o.noise_scale = std::atof(next());
    else if (a == "--seek") o.seek = std::atof(next());
    else if (a == "--pitch-gain") o.pitch_gain = std::atof(next());
    else if (a == "--pitch-limit") o.pitch_limit = std::atof(next());
    else if (a == "--height-offset") o.height_offset = o.height_offset_down = std::atof(next());
    else if (a == "--height-offset-down") o.height_offset_down = std::atof(next());
    else if (a == "--torque-margin") o.torque_margin = std::atof(next());
    else if (a == "--shift-tol") o.shift_tol = std::atof(next());
    else throw std::runtime_error("unknown argument " + a);
  }
  const std::vector<std::string> valid_gaits = {
      "stand", "trot", "flying_trot", "bound", "pace", "pronk", "walk", "gallop",
      "stair_trot", "stair_walk", "stair_crawl"};
  bool valid = false;
  for (const auto& g : valid_gaits) {
    if (o.gait == g) { valid = true; break; }
  }
  if (!valid) throw std::runtime_error("--gait: stand|trot|flying_trot|bound|pace|pronk|walk|gallop|stair_trot|stair_walk|stair_crawl");
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

// Per-MPC-tick CSV for analyze_log.py (enabled with MPC_LOG=<path>).
// <leg>_sat is a bitmask of saturated joints: 1 hip, 2 thigh, 4 calf.
class CsvLog {
 public:
  CsvLog(const mjModel* m, const char* path) : m_(m) {
    f_ = std::fopen(path, "w");
    if (!f_) throw std::runtime_error(std::string("cannot open ") + path);
    const char* legs[kNumLegs] = {"FR", "FL", "RR", "RL"};
    std::fprintf(f_, "t,gait,roll,pitch,yaw,cx,cy,cz,vx,vy,vz");
    for (int leg = 0; leg < kNumLegs; ++leg) {
      foot_body_[leg] = mj_name2id(m, mjOBJ_BODY, (std::string(legs[leg]) + "_foot").c_str());
      for (const char* c : {"sched", "cn", "fz_foot", "fx", "fy", "fz", "sat", "px"})
        std::fprintf(f_, ",%s_%s", legs[leg], c);
    }
    std::fprintf(f_, ",other_contacts,qp_ok,qp_iter\n");
    terrain_body_ = mj_name2id(m, mjOBJ_BODY, "terrain");
  }
  ~CsvLog() { std::fclose(f_); }

  // `tau` is the commanded joint torque, `limit` the motor envelope at that tick.
  void Write(const mjData* d, const LocomotionController& c, const RobotState& s,
             const Vec12& tau, const Vec12& limit) {
    std::array<double, kNumLegs> cn{};
    int other = 0;
    for (int i = 0; i < d->ncon; ++i) {
      const mjContact& con = d->contact[i];
      const int b1 = m_->geom_bodyid[con.geom1], b2 = m_->geom_bodyid[con.geom2];
      const int robot = IsGround(b1) ? b2 : IsGround(b2) ? b1 : -1;
      if (robot < 0) continue;
      int leg = -1;
      for (int l = 0; l < kNumLegs; ++l)
        if (robot == foot_body_[l]) leg = l;
      if (leg < 0) { ++other; continue; }
      mjtNum f[6];
      mj_contactForce(m_, d, i, f);
      cn[leg] += f[0];
    }
    const Vec12& f = c.forces();
    std::fprintf(f_, "%.3f,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", d->time,
                 c.gait().name, s.rpy.x(), s.rpy.y(), s.rpy.z(), s.com.x(), s.com.y(), s.com.z(),
                 s.vel_world.x(), s.vel_world.y(), s.vel_world.z());
    for (int leg = 0; leg < kNumLegs; ++leg) {
      int sat = 0;
      for (int j = 0; j < 3; ++j) {
        const int a = 3 * leg + j;  // actuators are ordered like the controller output
        if (std::abs(tau(a)) >= limit(a) - 1e-9) sat |= 1 << j;
      }
      std::fprintf(f_, ",%d,%.1f,%.4f,%.1f,%.1f,%.1f,%d,%.3f", c.stance()[leg] ? 1 : 0, cn[leg],
                   s.foot_world[leg].z(), f(3 * leg), f(3 * leg + 1), f(3 * leg + 2), sat,
                   s.foot_world[leg].x());
    }
    const MpcSolution& sol = c.last_solution();
    std::fprintf(f_, ",%d,%d,%d\n", other, sol.ok ? 1 : 0, sol.iterations);
  }

 private:
  bool IsGround(int body) const { return body == 0 || body == terrain_body_; }
  const mjModel* m_;
  std::FILE* f_ = nullptr;
  int terrain_body_ = -1;
  std::array<int, kNumLegs> foot_body_{};
};

class Simulation {
 public:
  Simulation(mjModel* m, const Options& opt) : m_(m), opt_(opt) {
    m_->opt.timestep = 0.001;  // 1000 Hz physics
    d_ = mj_makeData(m_);
    const int key = mj_name2id(m_, mjOBJ_KEY, "stand");
    if (key >= 0) mj_resetDataKeyframe(m_, d_, key);
    mj_forward(m_, d_);
    robot_ = std::make_unique<MujocoRobot>(m_, d_);
    std::array<int, 12> dof, act;
    for (int leg = 0; leg < kNumLegs; ++leg)
      for (int j = 0; j < 3; ++j) {
        dof[3 * leg + j] = robot_->dof(leg, j);
        act[3 * leg + j] = robot_->actuator(leg, j);
      }
    actuators_ = std::make_unique<ActuatorModel>(
        m_, dof, act, opt.knee_csv, opt.motor_caps ? JointCaps::kMotor : JointCaps::kUrdf,
        opt.torque_scale);

    if (opt.sensed_contact) {
      SensorNoise noise;
      noise.scale = opt.noise_scale;
      Eigen::VectorXd damping = Eigen::Map<const Eigen::VectorXd>(m_->dof_damping, m_->nv);
      estimator_ = std::make_unique<ContactEstimator>(dof, damping, 0.001, noise);
    }

    ControllerConfig cfg;
    cfg.physics_dt = m_->opt.timestep;
    cfg.ticks_per_mpc = static_cast<int>(std::lround(cfg.mpc.dt / cfg.physics_dt));
    double mass = 0.0;
    for (int b = 0; b < m_->nbody; ++b) mass += m_->body_mass[b];
    cfg.mpc.mass = mass;
    cfg.mpc.inertia_body = MpcParams::BoxInertia(mass, 0.70, 0.34, 0.28);
    terrain_ = std::make_unique<Terrain>(m_, d_);
    if (!terrain_->flat()) {  // hold the body level sideways and follow the slope
      cfg.mpc.state_weights[0] = opt.stair_roll_weight;
      cfg.mpc.state_weights[1] = opt.stair_pitch_weight;
      cfg.mpc.state_weights[3] = cfg.mpc.state_weights[4] = opt.stair_pos_weight;
    }
    cfg.stair.tread_margin = opt.tread_margin;
    cfg.stair.clearance = opt.stair_clearance;
    cfg.stair.contact_gating = opt.contact_gating;
    cfg.stair.gate_lift = opt.gate_lift;
    cfg.stair.shift_tolerance = opt.shift_tol;
    cfg.stair.thigh_torque_budget = opt.thigh_budget > 0.0 ? opt.thigh_budget : 1e9;
    cfg.stair.torque_aware = opt.torque_aware;
    cfg.stair.standoff_min = opt.standoff;
    cfg.stair.seek_kp = opt.seek;
    cfg.stair.pitch_gain = opt.pitch_gain;
    cfg.stair.pitch_limit = opt.pitch_limit;
    cfg.stair.height_offset = opt.height_offset;
    cfg.stair.height_offset_down = opt.height_offset_down;
    cfg.stair.torque_margin = opt.torque_margin;
    cfg.stair.leg_model = [this](int leg, const Vec3& foot, Mat3* J, Vec3* limit) {
      Vec3 q;
      if (!robot_->LegAt(leg, foot, J, &q)) return false;
      *limit = actuators_->StaticLegLimit(q);
      return true;
    };
    controller_ = std::make_unique<LocomotionController>(cfg, robot_->Read(), terrain_.get());
    controller_->SetCommand({0, 0, 0});
    if (const char* path = std::getenv("MPC_LOG")) log_ = std::make_unique<CsvLog>(m_, path);
  }
  ~Simulation() { mj_deleteData(d_); }

  bool Step() {
    if (tick_ == kStandTicks) {
      controller_->SetGait(ResolveGait(opt_.gait), tick_);
      controller_->SetCommand(opt_.cmd);
    }
    actuators_->PreStep(d_);  // knee reflected inertia at the current angle
    mj_step1(m_, d_);
    RobotState s = robot_->Read();
    if (estimator_) {
      const int nv = m_->nv;
      const auto fz = estimator_->Update(s, Eigen::Map<const Eigen::VectorXd>(d_->qvel, nv),
                                         Eigen::Map<const Eigen::VectorXd>(d_->qacc, nv),
                                         Eigen::Map<const Eigen::VectorXd>(d_->qfrc_applied, nv),
                                         tau_applied_);
      for (int leg = 0; leg < kNumLegs; ++leg) {
        const bool truth = s.foot_force_true[leg] >= 8.0, est = fz[leg] >= 8.0;
        ++confusion_[2 * truth + est];
        s.foot_force[leg] = fz[leg];
      }
    }
    const Vec12 tau_cmd = controller_->Update(tick_, s);
    tau_applied_ = actuators_->Apply(d_, tau_cmd);
    if (log_ && tick_ % kLogEvery == 0)
      log_->Write(d_, *controller_, s, tau_cmd, actuators_->limit());
    mj_step2(m_, d_);
    ++tick_;
    last_ = s;
    max_x_ = std::max(max_x_, s.com.x());
    max_z_ = std::max(max_z_, s.com.z());
    return s.com.z() > 0.10 && std::abs(s.rpy.x()) < 0.8 && std::abs(s.rpy.y()) < 0.9;
  }

  // Motor-side load per joint type over the whole run, against the manuals.
  // Stair progress: how far and how high the CoM got, against the terrain.
  void PrintContactSummary() const {
    if (!estimator_) return;
    const double n = confusion_[0] + confusion_[1] + confusion_[2] + confusion_[3];
    if (n <= 0.0) return;
    std::printf("estimated contact (>= 8 N) vs simulator: agree %.1f %%, false contact %.1f %%, "
                "missed contact %.1f %%\n",
                100.0 * (confusion_[0] + confusion_[3]) / n, 100.0 * confusion_[1] / n,
                100.0 * confusion_[2] / n);
  }

  void PrintCourseSummary() const {
    if (terrain_->flat()) return;
    const double top = terrain_->TopHeight();
    const double edge_first = terrain_->edges().front(), edge_last = terrain_->edges().back();
    int risers_cleared = 0;
    for (double e : terrain_->edges())
      if (max_x_ > e + 0.15 && terrain_->Height(e + 0.01) > terrain_->Height(e - 0.01)) ++risers_cleared;
    std::printf("course: terrain top %.2f m, stairs span x=%.2f..%.2f m | CoM reached x=%.2f m, "
                "z=%.2f m (top %s), risers climbed: %d\n",
                top, edge_first, edge_last, max_x_, max_z_,
                max_z_ > top + 0.20 ? "REACHED" : "not reached", risers_cleared);
    std::printf("gait clock: waited %.1f s for contact/weight shift, %d early touchdowns\n",
                controller_->hold_ticks() * 0.001, controller_->early_touchdowns());
  }

  void PrintActuatorSummary() const {
    static const char* kPart[3] = {"hip   (RS02)", "thigh (RS06)", "calf  (RS06)"};
    std::printf("actuators (motor output side; peak / rated torque, peak speed vs T-N end, "
                "share of ticks the envelope cut the command):\n");
    for (int part = 0; part < 3; ++part) {
      double peak = 0, sq = 0, speed = 0;
      long limited = 0, ticks = 0;
      for (int leg = 0; leg < kNumLegs; ++leg) {
        const ActuatorStats& st = actuators_->stats()[3 * leg + part];
        peak = std::max(peak, st.torque_peak);
        speed = std::max(speed, st.speed_peak);
        sq += st.torque_sq_sum;
        limited += st.limited_ticks;
        ticks += st.ticks;
      }
      const MotorSpec& ms = actuators_->motor(part);
      std::printf("  %s peak %5.1f/%4.1f N m (rms %4.1f)  speed %5.1f rad/s (%3.0f%% of %.1f)  "
                  "cut %4.1f%%\n",
                  kPart[part], peak, ms.peak_torque, std::sqrt(sq / std::max(1L, ticks)), speed,
                  100.0 * speed / ms.tn_curve.back().first, ms.tn_curve.back().first,
                  100.0 * limited / std::max(1L, ticks));
    }
  }

  void Print() const {
    const MpcSolution& sol = controller_->last_solution();
    const Vec12& f = controller_->forces();
    std::printf("t=%6.2fs gait=%-5s com=(%6.3f %6.3f %5.3f) rpy=(%6.3f %6.3f %6.3f) "
                "vx=%6.3f sum_fz=%6.1fN | QP %s it=%4d %5.2fms\n",
                d_->time, controller_->gait().name, last_.com.x(), last_.com.y(),
                last_.com.z(), last_.rpy.x(), last_.rpy.y(), last_.rpy.z(), last_.vel_world.x(),
                f(2) + f(5) + f(8) + f(11), sol.ok ? "ok " : "ERR", sol.iterations, sol.solve_ms);
  }

  mjModel* m() { return m_; }
  mjData* d() { return d_; }
  long tick() const { return tick_; }
  int base_body() const { return robot_->base_body(); }

 private:
  static constexpr long kStandTicks = 500;  // 0.5 s
  static constexpr long kLogEvery = 33;     // one row per MPC solve
  mjModel* m_;
  mjData* d_;
  Options opt_;
  std::unique_ptr<MujocoRobot> robot_;
  std::unique_ptr<LocomotionController> controller_;
  std::unique_ptr<Terrain> terrain_;
  std::unique_ptr<ActuatorModel> actuators_;
  std::unique_ptr<ContactEstimator> estimator_;
  Vec12 tau_applied_ = Vec12::Zero();
  long confusion_[4] = {0, 0, 0, 0};  // [2 * truth + estimate] ticks per foot
  double max_x_ = -1e9, max_z_ = -1e9;
  std::unique_ptr<CsvLog> log_;
  RobotState last_;
  long tick_ = 0;
};

int RunHeadless(Simulation& sim, double duration) {
  const auto start = std::chrono::steady_clock::now();
  while (sim.d()->time < duration) {
    if (!sim.Step()) {
      sim.Print();
      std::printf("FALLEN at t=%.3f s\n", sim.d()->time);
      sim.PrintCourseSummary();
      sim.PrintContactSummary();
      sim.PrintActuatorSummary();
      return 1;
    }
    if (sim.tick() % 500 == 0) sim.Print();
  }
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("Finished %.1f s of simulation in %.2f s wall (%.1fx real time)\n", duration, wall,
              duration / wall);
  sim.PrintCourseSummary();
  sim.PrintContactSummary();
  sim.PrintActuatorSummary();
  return 0;
}

#ifdef ROBOST_MPC_WITH_VIEWER
// Mouse / keyboard camera control for the viewer.
struct ViewState {
  const mjModel* m = nullptr;
  mjvCamera* cam = nullptr;
  mjvScene* scn = nullptr;
  const mjData* d = nullptr;
  int base_body = 0;
  bool left = false, middle = false, right = false;
  double last_x = 0.0, last_y = 0.0;
  bool paused = false;
  bool help = true;
};

// Switch to the free camera, starting where the tracking camera is looking.
void FreeCamera(ViewState* v) {
  if (v->cam->type == mjCAMERA_FREE) return;
  for (int i = 0; i < 3; ++i) v->cam->lookat[i] = v->d->xpos[3 * v->base_body + i];
  v->cam->type = mjCAMERA_FREE;
}

void SetView(ViewState* v, double azimuth, double elevation, double distance) {
  v->cam->azimuth = azimuth;
  v->cam->elevation = elevation;
  v->cam->distance = distance;
}

void OnKey(GLFWwindow* w, int key, int, int action, int) {
  if (action != GLFW_PRESS) return;
  auto* v = static_cast<ViewState*>(glfwGetWindowUserPointer(w));
  switch (key) {
    case GLFW_KEY_SPACE: v->paused = !v->paused; break;
    case GLFW_KEY_H: v->help = !v->help; break;
    case GLFW_KEY_T:  // follow the robot <-> free camera
      if (v->cam->type == mjCAMERA_FREE) {
        v->cam->type = mjCAMERA_TRACKING;
        v->cam->trackbodyid = v->base_body;
      } else {
        FreeCamera(v);
      }
      break;
    case GLFW_KEY_1: SetView(v, 135, -20, 2.0); break;   // oblique (default)
    case GLFW_KEY_2: SetView(v, 90, -5, 2.5); break;     // side
    case GLFW_KEY_3: SetView(v, 0, -5, 2.5); break;      // seen from behind
    case GLFW_KEY_4: SetView(v, 180, -5, 2.5); break;    // seen from the front
    case GLFW_KEY_5: SetView(v, 90, -89, 3.5); break;    // top
    case GLFW_KEY_EQUAL: v->cam->distance *= 0.85; break;
    case GLFW_KEY_MINUS: v->cam->distance *= 1.15; break;
    default: break;
  }
}

void OnMouseButton(GLFWwindow* w, int button, int action, int) {
  auto* v = static_cast<ViewState*>(glfwGetWindowUserPointer(w));
  const bool down = action == GLFW_PRESS;
  if (button == GLFW_MOUSE_BUTTON_LEFT) v->left = down;
  if (button == GLFW_MOUSE_BUTTON_MIDDLE) v->middle = down;
  if (button == GLFW_MOUSE_BUTTON_RIGHT) v->right = down;
  glfwGetCursorPos(w, &v->last_x, &v->last_y);
}

void OnMouseMove(GLFWwindow* w, double x, double y) {
  auto* v = static_cast<ViewState*>(glfwGetWindowUserPointer(w));
  if (!v->left && !v->middle && !v->right) return;
  const double dx = x - v->last_x, dy = y - v->last_y;
  v->last_x = x;
  v->last_y = y;
  int width, height;
  glfwGetWindowSize(w, &width, &height);
  const bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  mjtMouse action;
  if (v->right) {  // pan
    FreeCamera(v);
    action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (v->left) {  // orbit
    action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {  // zoom
    action = mjMOUSE_ZOOM;
  }
  mjv_moveCamera(v->m, action, dx / height, dy / height, v->cam);
}

void OnScroll(GLFWwindow* w, double, double yoffset) {
  auto* v = static_cast<ViewState*>(glfwGetWindowUserPointer(w));
  mjv_moveCamera(v->m, mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, v->cam);
}

int RunViewer(Simulation& sim, const Options& opt) {
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

  ViewState view;
  view.m = sim.m();
  view.d = sim.d();
  view.cam = &cam;
  view.scn = &scn;
  view.base_body = sim.base_body();
  glfwSetWindowUserPointer(window, &view);
  glfwSetKeyCallback(window, OnKey);
  glfwSetMouseButtonCallback(window, OnMouseButton);
  glfwSetCursorPosCallback(window, OnMouseMove);
  glfwSetScrollCallback(window, OnScroll);

  bool fallen = false;
  auto wall_start = std::chrono::steady_clock::now();
  while (!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count() * opt.speed;
    int steps_this_frame = 0;
    while (!view.paused && !fallen && sim.d()->time < wall && steps_this_frame < 30) {
      if (!sim.Step()) {
        fallen = true;
        sim.Print();
        std::printf("FALLEN at t=%.3f s\n", sim.d()->time);
      }
      if (sim.tick() % 500 == 0) sim.Print();
      ++steps_this_frame;
    }
    if (wall - sim.d()->time > 0.05) {
      wall_start = std::chrono::steady_clock::now() -
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double>(sim.d()->time / opt.speed));
    }
    if (opt.duration > 0 && sim.d()->time > opt.duration) break;

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(sim.m(), sim.d(), &vopt, nullptr, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    char info[128];
    std::snprintf(info, sizeof(info), "%.2f s%s%s", sim.d()->time, fallen ? "  FALLEN" : "",
                  view.paused ? "  PAUSED" : "");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, "time", info, &con);
    if (view.help)
      mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport,
                  "mouse left: orbit (shift: roll)\nmouse right: pan (shift: other plane)\n"
                  "wheel / middle: zoom (also +/-)\nT: follow <-> free camera\n"
                  "1 oblique  2 side  3 rear  4 front  5 top\nSpace: pause   H: hide help   Esc: quit",
                  "", &con);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  mjr_freeContext(&con);
  mjv_freeScene(&scn);
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
    Simulation sim(m, opt);
    int rc = 0;
    std::printf("RS06 Convex MPC | gait=%s cmd=(%.2f, %.2f, %.2f) | physics 1000 Hz, MPC 30 Hz\n",
                opt.gait.c_str(), opt.cmd.vx, opt.cmd.vy, opt.cmd.yaw_rate);
#ifdef ROBOST_MPC_WITH_VIEWER
    rc = opt.headless ? RunHeadless(sim, opt.duration) : RunViewer(sim, opt);
#else
    rc = RunHeadless(sim, opt.duration > 0 ? opt.duration : 10.0);
#endif
    mj_deleteModel(m);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
