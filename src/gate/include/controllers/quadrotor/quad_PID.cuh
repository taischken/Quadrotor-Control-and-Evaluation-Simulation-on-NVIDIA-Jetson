#ifndef QUAD_PID_CUH_
#define QUAD_PID_CUH_

/*
 * Cascaded PID quadrotor controller with PER-ROLLOUT GAINS and
 * SMOOTH REFERENCE TRACKING.
 *
 * Now uses a time-varying reference computed by mission_reference.cuh
 * instead of step-jumping between waypoints. This lets the controller
 * track long-distance waypoints without saturating its tilt commands.
 *
 * Architecture (unchanged):
 *   OUTER LOOP (position)
 *     pid_z   : altitude error -> thrust correction
 *     pid_x   : x error        -> desired pitch  theta_d
 *     pid_y   : y error        -> desired roll   phi_d  (negated)
 *
 *   INNER LOOP (attitude)
 *     pid_phi   : roll  error  -> tau_phi
 *     pid_theta : pitch error  -> tau_theta
 *     pid_psi   : yaw   error  -> tau_psi
 *
 * Output: u = [U_total, tau_phi, tau_theta, tau_psi]  (CONTROL_DIM = 4)
 */

#pragma once
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <cuda_util/stream_managed.cuh>
#include <controllers/guidance_stream_managed.cuh>
#include <dynamics/dynamics_stream_managed.cuh>
#include <dynamics/util.cuh>
#include <cuda_util/cuda_memory_utils.cuh>
#include <mission_reference.cuh>     // <-- NEW: shared reference math
#include <random>
#include <vector>

namespace quad_pid {
    const int S_DIM = 12;
    const int C_DIM = 4;
    const int N_PID = 6;
}

struct PIDGainBlock {
    float kp[quad_pid::N_PID];
    float ki[quad_pid::N_PID];
    float kd[quad_pid::N_PID];
    float windup[quad_pid::N_PID];
};

struct PIDStateBlock {
    float integral[quad_pid::N_PID];
    float prev_measured[quad_pid::N_PID];
    int   first_run[quad_pid::N_PID];
};

inline PIDGainBlock nominal_gains() {
    PIDGainBlock g{};
    g.kp[0] = 5.6f;  g.ki[0] = 0.5f;   g.kd[0] = 5.6f;  g.windup[0] = 30.0f;
    g.kp[1] = 0.23f; g.ki[1] = 0.005f; g.kd[1] = 0.31f; g.windup[1] = 1.0f;
    g.kp[2] = 0.23f; g.ki[2] = 0.005f; g.kd[2] = 0.31f; g.windup[2] = 1.0f;
    g.kp[3] = 18.0f; g.ki[3] = 0.0f;   g.kd[3] = 0.9f;  g.windup[3] = 10.0f;
    g.kp[4] = 18.0f; g.ki[4] = 0.0f;   g.kd[4] = 0.9f;  g.windup[4] = 10.0f;
    g.kp[5] = 37.0f; g.ki[5] = 0.0f;   g.kd[5] = 1.8f;  g.windup[5] = 10.0f;
    return g;
}

/*
 * Static guidance params. Now CARRIES the mission, instead of having
 * setpoints baked in. The mission is the single source of truth.
 */
struct QuadPIDParams {
    MissionRef mission;       // <-- NEW: shared mission definition

    float angle_limit = 0.44f;
    float dt = 0.005f;

    QuadPIDParams()  = default;
    ~QuadPIDParams() = default;
};

template<int NUM_ROLLOUTS>
class QuadPID : public GATE_internal::Guidance<QuadPID<NUM_ROLLOUTS>,
                                               QuadPIDParams, QuadParams,
                                               quad_pid::S_DIM, quad_pid::C_DIM> {
public:
    using Base = GATE_internal::Guidance<QuadPID<NUM_ROLLOUTS>, QuadPIDParams, QuadParams,
                                         quad_pid::S_DIM, quad_pid::C_DIM>;
    using state_array   = typename Base::state_array;
    using control_array = typename Base::control_array;

    QuadPID(cudaStream_t stream = nullptr) : Base(stream) {
        this->guidance_params_ = QuadPIDParams();
        this->dynamics_params_ = QuadParams();
        allocateCUDAMem();
    }
    ~QuadPID() {
        deallocateCudaMem();
    }

    PIDGainBlock*  pid_gains_d_  = nullptr;
    PIDStateBlock* pid_states_d_ = nullptr;
    std::vector<PIDGainBlock> host_gains;

    void sampleGainsUniform(const PIDGainBlock& nominal, float spread = 0.5f,
                            unsigned int seed = 0xC0FFEE)
    {
        host_gains.assign(NUM_ROLLOUTS, PIDGainBlock{});
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(1.0f - spread, 1.0f + spread);
        host_gains[0] = nominal;
        for (int r = 1; r < NUM_ROLLOUTS; ++r) {
            for (int p = 0; p < quad_pid::N_PID; ++p) {
                host_gains[r].kp[p]     = nominal.kp[p] * dist(rng);
                host_gains[r].ki[p]     = nominal.ki[p] * dist(rng);
                host_gains[r].kd[p]     = nominal.kd[p] * dist(rng);
                host_gains[r].windup[p] = nominal.windup[p];
            }
        }
        HANDLE_ERROR(cudaMemcpyAsync(pid_gains_d_, host_gains.data(),
                                     sizeof(PIDGainBlock) * NUM_ROLLOUTS,
                                     cudaMemcpyHostToDevice, this->stream_));
        HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
        CudaCheckError();
    }

    void initPIDStates() {
        std::vector<PIDStateBlock> host_init(NUM_ROLLOUTS);
        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            for (int p = 0; p < quad_pid::N_PID; ++p) {
                host_init[r].integral[p]      = 0.0f;
                host_init[r].prev_measured[p] = 0.0f;
                host_init[r].first_run[p]     = 1;
            }
        }
        HANDLE_ERROR(cudaMemcpyAsync(pid_states_d_, host_init.data(),
                                     sizeof(PIDStateBlock) * NUM_ROLLOUTS,
                                     cudaMemcpyHostToDevice, this->stream_));
        HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
        CudaCheckError();
    }

    __host__ __device__ float pid_step(int pid_idx, float setpoint, float meas,
                                       float dt_, int tid)
    {
        PIDGainBlock&  g  = pid_gains_d_[tid];
        PIDStateBlock& st = pid_states_d_[tid];

        float kp = g.kp[pid_idx], ki = g.ki[pid_idx], kd = g.kd[pid_idx];
        float wl = g.windup[pid_idx];

        float error = setpoint - meas;
        st.integral[pid_idx] += error * dt_;
        if (st.integral[pid_idx] >  wl) st.integral[pid_idx] =  wl;
        if (st.integral[pid_idx] < -wl) st.integral[pid_idx] = -wl;

        float deriv = 0.0f;
        if (st.first_run[pid_idx] == 0)
            deriv = -(meas - st.prev_measured[pid_idx]) / dt_;
        st.first_run[pid_idx]     = 0;
        st.prev_measured[pid_idx] = meas;

        return kp * error + ki * st.integral[pid_idx] + kd * deriv;
    }

    template<class PER_T>
    __host__ __device__ void getControl(PER_T* pert, int timestep, int rollout,
                                        int num_rollouts,
                                        state_array* x, state_array* xdot,
                                        control_array* u_k, control_array* u_kp1)
    {
        const QuadPIDParams& gp = this->guidance_params_;
        const QuadParams&    dp = this->dynamics_params_;
        const float dt_ = gp.dt;

        float px = (*x)(0), py = (*x)(1), pz = (*x)(2);
        float phi = (*x)(6), theta = (*x)(7), psi = (*x)(8);

        // ---- Smooth reference from the shared mission ----
        float des_x, des_y, des_z;
        compute_reference(timestep, gp.mission, des_x, des_y, des_z);
        float des_psi = gp.mission.des_psi;

        // ---- Outer loop (position) ----
        float U = dp.m * dp.g + pid_step(0, des_z, pz, dt_, rollout);
        float theta_d =  pid_step(1, des_x, px, dt_, rollout);
        float phi_d   = -pid_step(2, des_y, py, dt_, rollout);

        if (theta_d >  gp.angle_limit) theta_d =  gp.angle_limit;
        if (theta_d < -gp.angle_limit) theta_d = -gp.angle_limit;
        if (phi_d   >  gp.angle_limit) phi_d   =  gp.angle_limit;
        if (phi_d   < -gp.angle_limit) phi_d   = -gp.angle_limit;

        // ---- Inner loop (attitude) ----
        float tau_phi   = pid_step(3, phi_d,   phi,   dt_, rollout);
        float tau_theta = pid_step(4, theta_d, theta, dt_, rollout);
        float tau_psi   = pid_step(5, des_psi, psi,   dt_, rollout);

        (*u_k)(0) = U;
        (*u_k)(1) = tau_phi;
        (*u_k)(2) = tau_theta;
        (*u_k)(3) = tau_psi;

        if (rollout > 0) {
            control_array input_pert;
            input_pert.block<quad_pid::C_DIM, 1>(0, 0) <<
                (*(pert->u_t_pert_d_)).block<quad_pid::C_DIM, 1>(
                    0, num_rollouts * timestep + rollout);
            (*u_k) += input_pert;
        }
        *u_kp1 = *u_k;
    }

private:
    void allocateCUDAMem() {
        if (pid_gains_d_  == nullptr)
            HANDLE_ERROR(cudaMalloc((void**)&pid_gains_d_,
                                    sizeof(PIDGainBlock)  * NUM_ROLLOUTS));
        if (pid_states_d_ == nullptr)
            HANDLE_ERROR(cudaMalloc((void**)&pid_states_d_,
                                    sizeof(PIDStateBlock) * NUM_ROLLOUTS));
        CudaCheckError();
    }
    void deallocateCudaMem() {
        if (pid_gains_d_)  { cudaFree(pid_gains_d_);  pid_gains_d_  = nullptr; }
        if (pid_states_d_) { cudaFree(pid_states_d_); pid_states_d_ = nullptr; }
    }
};

#endif // QUAD_PID_CUH_
