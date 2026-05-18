#include <trajectory_generation/trajectory_export.h>
#include <trajectory_generation/GATE.cuh>
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <controllers/quadrotor/quad_PID.cuh>
#include <perturbations/quadrotor/quad_perturbations.cuh>
#include <dynamics/integrators_eigen.cuh>
#include <scoring/scoring_host.h>
#include <memory>
#include <iostream>
#include <cmath>

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>

#include "cpu_simulator.h"
#include "viewer.h"
#include "mission_reference.cuh"


/*
 * Real-time application with SMOOTH reference trajectories.
 *
 *   1. User inputs mission waypoints
 *   2. Host computes phase boundaries (scales with mission distance)
 *   3. GPU rollout kernel runs 1024 perturbed-gain rollouts
 *   4. GPU scoring kernel ranks them
 *   5. Best gains are extracted
 *   6. CPU simulator runs with those gains, alongside GPU replay
 *   7. Viewer animates both
 *
 * Mission is shared via mission_reference.cuh -- single source of truth.
 */

constexpr int NUM_TIMESTEPS = 20000;   // 100 s at dt=0.005 (was 8000 = 40s)
constexpr int NUM_ROLLOUTS  = 1024;
constexpr int BDIM_X        = 64;


struct MissionInput {
    float z1;
    float x_target, y_target;
    float z2;
};


MissionInput prompt_user_for_mission() {
    MissionInput m;
    std::cout << "\n===================================================\n"
              << "       Real-Time Drone Mission Configuration       \n"
              << "===================================================\n\n";

    std::cout << "Enter z1 (first hover altitude, meters)         : ";
    std::cin  >> m.z1;
    std::cout << "Enter target x (meters)                         : ";
    std::cin  >> m.x_target;
    std::cout << "Enter target y (meters)                         : ";
    std::cin  >> m.y_target;
    std::cout << "Enter z2 (final altitude after x-y move, meters): ";
    std::cin  >> m.z2;

    return m;
}


int find_best_rollout(const scoring::ScoreResults& s) {
    int best = -1;
    float best_cost = 1e30f;
    for (size_t r = 0; r < s.combined.size(); ++r) {
        if (s.crashed[r]) continue;
        if (s.combined[r] < best_cost) {
            best_cost = s.combined[r];
            best = (int)r;
        }
    }
    if (best < 0) {
        std::cerr << "ERROR: every rollout crashed!\n";
        std::exit(1);
    }
    return best;
}


cpu_sim::PIDGains gpu_gains_to_cpu(const PIDGainBlock& g) {
    cpu_sim::PIDGains out;
    for (int i = 0; i < 6; ++i) {
        out.kp[i]     = g.kp[i];
        out.ki[i]     = g.ki[i];
        out.kd[i]     = g.kd[i];
        out.windup[i] = g.windup[i];
    }
    return out;
}


int main() {

    MissionInput user_mission = prompt_user_for_mission();

    float dt = 0.005f;

    // ============ Build the shared mission ============
    MissionRef mission;
    mission.z1       = user_mission.z1;
    mission.x_target = user_mission.x_target;
    mission.y_target = user_mission.y_target;
    mission.z2       = user_mission.z2;
    mission.dt       = dt;

    // Auto-scale phase boundaries based on horizontal distance
    compute_phase_boundaries(mission, NUM_TIMESTEPS, /*cruise=*/5.0f);

    float horiz_dist = sqrtf(user_mission.x_target * user_mission.x_target
                              + user_mission.y_target * user_mission.y_target);
    std::cout << "\nMission accepted:\n"
              << "  Horizontal distance: " << horiz_dist << " m\n"
              << "  Phase 1 (climb)     : 0..." << mission.step_end_1 * dt << " s\n"
              << "  Phase 2 (translate) : " << mission.step_end_1 * dt
                                      << "..." << mission.step_end_2 * dt << " s\n"
              << "  Phase 3 (altitude)  : " << mission.step_end_2 * dt
                                      << "..." << mission.step_end_3 * dt << " s\n"
              << "  Phase 4 (settle)    : " << mission.step_end_3 * dt
                                      << "..." << NUM_TIMESTEPS * dt << " s\n\n";

    // ============ Build the GATE pipeline ============
    std::shared_ptr<QuadDynamics> dyn = std::make_shared<QuadDynamics>();
    std::shared_ptr<QuadPID<NUM_ROLLOUTS>> ctrl =
        std::make_shared<QuadPID<NUM_ROLLOUTS>>();

    auto guid_params = ctrl->getGuidanceParams();
    auto dyn_params  = ctrl->getDynamicsParams();
    guid_params.dt       = dt;
    guid_params.mission  = mission;       // share the mission
    ctrl->setGuidanceParams(guid_params);
    ctrl->setDynamicsParams(dyn_params);
    dyn->setParams(dyn_params);

    // Perturbations: zero physics noise (gain effect only)
    QuadPertParams::state_array   x0_mean, x0_std;
    QuadPertParams::control_array u_std;
    x0_mean.setZero(); x0_std.setZero(); u_std.setZero();
    QuadPertParams pert_params(x0_mean, x0_std, u_std);
    pert_params.m_std_ = pert_params.ixx_std_ = 0.0f;
    pert_params.iyy_std_ = pert_params.izz_std_ = 0.0f;

    std::shared_ptr<QuadPert<NUM_TIMESTEPS, NUM_ROLLOUTS>> pert =
        std::make_shared<QuadPert<NUM_TIMESTEPS, NUM_ROLLOUTS>>(pert_params);
    pert->initializeX0andControlPerturbations();
    pert->initPerturbations();

    typedef GATE<QuadDynamics, QuadPID<NUM_ROLLOUTS>,
                 QuadPert<NUM_TIMESTEPS, NUM_ROLLOUTS>,
                 NUM_TIMESTEPS, NUM_ROLLOUTS, BDIM_X> Quad_ROTE;
    std::shared_ptr<Quad_ROTE> RTE =
        std::make_shared<Quad_ROTE>(dyn.get(), ctrl.get(), pert.get(),
                                    x0_mean, dt);

    PIDGainBlock nominal = nominal_gains();
    ctrl->sampleGainsUniform(nominal, /*spread=*/0.5f, /*seed=*/0xC0FFEE);
    ctrl->initPIDStates();

    // ============ GPU rollout ============
    std::cout << "Running GPU rollout kernel..." << std::flush;
    RTE->computeTrajectories();
    std::cout << " done.\n";

    // ============ GPU scoring ============
    std::cout << "Running GPU scoring kernel..." << std::flush;
    scoring::ScoringParams sp;
    sp.mission = mission;          // same mission for the scorer
    scoring::ScoreResults scores = scoring::run_scoring_kernel(
        RTE->state_trajectories_device,
        NUM_ROLLOUTS, NUM_TIMESTEPS, QuadDynamics::STATE_DIM, dt, sp);
    std::cout << " done.\n";

    int num_crashed = 0;
    for (int c : scores.crashed) num_crashed += c;
    std::cout << "Crashed rollouts: " << num_crashed << " / "
              << NUM_ROLLOUTS << "\n";

    // ============ Pick winner ============
    int best = find_best_rollout(scores);
    std::cout << "\nBest rollout: #" << best
              << "    combined cost = " << scores.combined[best]
              << "    RMSE = "         << scores.rmse[best]
              << "    overshoot = "    << scores.overshoot[best] << "\n";

    const PIDGainBlock& wg = ctrl->host_gains[best];
    std::cout << "Winning gains:\n";
    std::cout << "  kp_z     = " << wg.kp[0] << "   kd_z     = " << wg.kd[0] << "\n";
    std::cout << "  kp_x     = " << wg.kp[1] << "   kd_x     = " << wg.kd[1] << "\n";
    std::cout << "  kp_y     = " << wg.kp[2] << "   kd_y     = " << wg.kd[2] << "\n";
    std::cout << "  kp_phi   = " << wg.kp[3] << "   kd_phi   = " << wg.kd[3] << "\n";
    std::cout << "  kp_theta = " << wg.kp[4] << "   kd_theta = " << wg.kd[4] << "\n";
    std::cout << "  kp_psi   = " << wg.kp[5] << "   kd_psi   = " << wg.kd[5] << "\n";

    // ============ Extract winning trajectory ============
    std::vector<float> best_traj(NUM_TIMESTEPS * 12);
    const float* host_buf = RTE->state_trajectories_host.data();
    for (int k = 0; k < NUM_TIMESTEPS; ++k) {
        for (int i = 0; i < 12; ++i) {
            best_traj[k * 12 + i] =
                host_buf[best * 12 * NUM_TIMESTEPS + k * 12 + i];
        }
    }

    // ============ CPU simulator with winning gains ============
    cpu_sim::PIDGains cpu_gains = gpu_gains_to_cpu(wg);
    cpu_sim::PhysParams phys;
    cpu_sim::Simulator simulator(cpu_gains, mission, phys,
                                  /*angle_limit=*/0.44f, NUM_TIMESTEPS);

    // ============ Launch viewer ============
    ViewerData v;
    v.gpu_trajectory   = best_traj.data();
    v.gpu_num_steps    = NUM_TIMESTEPS;
    v.cpu_simulator    = &simulator;
    v.mission          = mission;
    v.dt               = dt;
    v.best_rollout_idx = best;
    v.playback_speed   = 4.0f;          // 4x = 100s sim in 25s real time

    std::cout << "\nOpening viewer... Press SPACE pause, R reset, ESC quit.\n";
    run_viewer(v);

    std::cout << "Done.\n";
    return 0;
}
