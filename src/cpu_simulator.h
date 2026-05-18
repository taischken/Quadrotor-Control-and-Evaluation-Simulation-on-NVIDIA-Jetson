#ifndef CPU_SIMULATOR_H_
#define CPU_SIMULATOR_H_

/*
 * CPU-side stateful quadrotor simulator -- now using the SHARED mission
 * reference. All reference math is in mission_reference.cuh, so the CPU
 * and GPU compute exactly the same desired (x,y,z) at every timestep.
 */

#include <cmath>
#include "mission_reference.cuh"

namespace cpu_sim {

    struct PhysParams {
        float m  = 1.4f;
        float g  = 9.8f;
        float Ix = 0.0116f;
        float Iy = 0.0116f;
        float Iz = 0.0232f;
    };

    struct PIDGains {
        float kp[6];
        float ki[6];
        float kd[6];
        float windup[6];
    };

    struct PIDStates {
        float integral[6]      = {0};
        float prev_measured[6] = {0};
        int   first_run[6]     = {1,1,1,1,1,1};
    };

    inline float pid_step(int idx,
                          float setpoint, float meas,
                          const PIDGains& g, PIDStates& st, float dt)
    {
        float error = setpoint - meas;
        st.integral[idx] += error * dt;
        if (st.integral[idx] >  g.windup[idx]) st.integral[idx] =  g.windup[idx];
        if (st.integral[idx] < -g.windup[idx]) st.integral[idx] = -g.windup[idx];

        float deriv = 0.0f;
        if (st.first_run[idx] == 0)
            deriv = -(meas - st.prev_measured[idx]) / dt;
        st.first_run[idx]     = 0;
        st.prev_measured[idx] = meas;

        return g.kp[idx] * error + g.ki[idx] * st.integral[idx] + g.kd[idx] * deriv;
    }

    inline void compute_control(int timestep,
                                 const float* x,
                                 const PIDGains& g, PIDStates& st,
                                 const MissionRef& mis, const PhysParams& phys,
                                 float angle_limit, float* u)
    {
        float px = x[0], py = x[1], pz = x[2];
        float phi = x[6], theta = x[7], psi = x[8];

        // SHARED reference
        float des_x, des_y, des_z;
        compute_reference(timestep, mis, des_x, des_y, des_z);

        float U = phys.m * phys.g + pid_step(0, des_z, pz, g, st, mis.dt);
        float theta_d =  pid_step(1, des_x, px, g, st, mis.dt);
        float phi_d   = -pid_step(2, des_y, py, g, st, mis.dt);

        if (theta_d >  angle_limit) theta_d =  angle_limit;
        if (theta_d < -angle_limit) theta_d = -angle_limit;
        if (phi_d   >  angle_limit) phi_d   =  angle_limit;
        if (phi_d   < -angle_limit) phi_d   = -angle_limit;

        float tau_phi   = pid_step(3, phi_d,   phi,   g, st, mis.dt);
        float tau_theta = pid_step(4, theta_d, theta, g, st, mis.dt);
        float tau_psi   = pid_step(5, mis.des_psi, psi, g, st, mis.dt);

        u[0] = U;
        u[1] = tau_phi;
        u[2] = tau_theta;
        u[3] = tau_psi;
    }

    inline void state_deriv(const float* x, const float* u,
                            const PhysParams& p, float* xdot)
    {
        float vx = x[3], vy = x[4], vz = x[5];
        float phi = x[6], theta = x[7], psi = x[8];
        float pp = x[9], qq = x[10], rr = x[11];

        float U         = u[0];
        float tau_phi   = u[1];
        float tau_theta = u[2];
        float tau_psi   = u[3];

        float sphi = sinf(phi),   cphi = cosf(phi);
        float sth  = sinf(theta), cth  = cosf(theta);
        float spsi = sinf(psi),   cpsi = cosf(psi);

        float ax =  (spsi*sphi + cpsi*sth*cphi) * U / p.m;
        float ay = (-cpsi*sphi + spsi*sth*cphi) * U / p.m;
        float az =   cth*cphi * U / p.m  -  p.g;

        float p_dot = ((p.Ix + p.Iz - p.Iy) * rr * qq + tau_phi)   / p.Ix;
        float q_dot = ((p.Iz - p.Ix - p.Iy) * pp * rr + tau_theta) / p.Iy;
        float r_dot = ((p.Ix + p.Iy - p.Iz) * pp * qq + tau_psi)   / p.Iz;

        xdot[0]  = vx;        xdot[1]  = vy;        xdot[2]  = vz;
        xdot[3]  = ax;        xdot[4]  = ay;        xdot[5]  = az;
        xdot[6]  = pp;        xdot[7]  = qq;        xdot[8]  = rr;
        xdot[9]  = p_dot;     xdot[10] = q_dot;     xdot[11] = r_dot;
    }

    inline void rk4_step(float* x, const float* u, const PhysParams& p, float dt)
    {
        float k1[12], k2[12], k3[12], k4[12], tmp[12];
        state_deriv(x, u, p, k1);
        for (int i=0;i<12;++i) tmp[i] = x[i] + 0.5f*dt*k1[i];
        state_deriv(tmp, u, p, k2);
        for (int i=0;i<12;++i) tmp[i] = x[i] + 0.5f*dt*k2[i];
        state_deriv(tmp, u, p, k3);
        for (int i=0;i<12;++i) tmp[i] = x[i] + dt*k3[i];
        state_deriv(tmp, u, p, k4);
        for (int i=0;i<12;++i)
            x[i] = x[i] + (dt/6.0f) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }

    class Simulator {
    public:
        Simulator(const PIDGains& g, const MissionRef& m, const PhysParams& p,
                  float angle_limit, int total_steps)
            : gains_(g), mission_(m), phys_(p), angle_limit_(angle_limit),
              total_steps_(total_steps), timestep_(0)
        {
            for (int i=0;i<12;++i) state_[i] = 0.0f;
        }

        const float* step() {
            float u[4];
            compute_control(timestep_, state_, gains_, pid_state_,
                            mission_, phys_, angle_limit_, u);
            rk4_step(state_, u, phys_, mission_.dt);
            timestep_++;
            return state_;
        }

        int current_step() const { return timestep_; }
        const float* state() const { return state_; }
        bool done() const { return timestep_ >= total_steps_; }
        const MissionRef& mission() const { return mission_; }
        int total_steps() const { return total_steps_; }

    private:
        PIDGains   gains_;
        MissionRef mission_;
        PhysParams phys_;
        float      angle_limit_;
        int        total_steps_;
        PIDStates  pid_state_;
        float      state_[12];
        int        timestep_;
    };

}   // namespace cpu_sim

#endif // CPU_SIMULATOR_H_
