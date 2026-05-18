#ifndef QUAD_DYNAMICS_CUH_
#define QUAD_DYNAMICS_CUH_

#include <dynamics/dynamics_stream_managed.cuh>
#include <perturbations/perturbations_stream_managed.cuh>
#include <perturbations/quadrotor/quad_perturbations.cuh>

using namespace Eigen;

/*
 * Quadrotor dynamics ported from quadrotor_controller_claude_2.cpp.
 *
 * STATE LAYOUT (12 elements, STANDARD aerospace convention):
 *    x[ 0..2]  = position    (px, py, pz)            [m]
 *    x[ 3..5]  = velocity    (vx, vy, vz)            [m/s]
 *    x[ 6..8]  = euler angles (phi=roll,
 *                              theta=pitch,
 *                              psi=yaw)              [rad]
 *    x[ 9..11] = body rates  (p, q, r)               [rad/s]
 *
 * CONTROL LAYOUT (4 elements):
 *    u[0] = U          total thrust   [N]
 *    u[1] = tau_phi    roll torque    [N*m]
 *    u[2] = tau_theta  pitch torque   [N*m]
 *    u[3] = tau_psi    yaw   torque   [N*m]
 *
 * Note on the remap from the user's cpp:
 *   The cpp file used psi=roll, theta=pitch, phi=yaw (non-standard).
 *   Here we rename to standard aerospace (phi=roll, theta=pitch, psi=yaw).
 *   The MATH is identical -- only variable names change.
 */

namespace quad {
    const int S_DIM = 12;
    const int C_DIM = 4;   // [U, tau_phi, tau_theta, tau_psi]
}

struct QuadParams {
    // Mass / geometry  (defaults match the cpp's main(): mass=1.4, L=1.0)
    float m         = 1.4f;     // quadrotor mass             [kg]
    float arm_len   = 1.0f;     // arm length                 [m]
    float prop_cons = 1e-6f;    // propeller force constant   [N / (rad/s)^2]
    float g         = 9.8f;     // gravity                    [m/s^2]

    // Diagonal inertia (standard convention: Ix = roll inertia)
    float Ix = 0.0116f;
    float Iy = 0.0116f;
    float Iz = 0.0232f;

    QuadParams()  = default;
    ~QuadParams() = default;
};

class QuadDynamics : public GATE_internal::Dynamics<QuadDynamics, QuadParams,
                                                     quad::S_DIM, quad::C_DIM> {
public:
    QuadDynamics(cudaStream_t stream = nullptr) :
        GATE_internal::Dynamics<QuadDynamics, QuadParams,
                                 quad::S_DIM, quad::C_DIM>(stream) {
        this->params_ = QuadParams();
    }
    ~QuadDynamics() = default;

    /*
     * f(x, u) = xdot. Stateless. Called 4x per RK4 step.
     */
    template<class PER_T>
    __host__ __device__ void quad_dynamics(PER_T* pert, int rollout,
                                           state_array& x,
                                           control_array& u,
                                           state_array& xdot)
    {
        // ---- Apply per-rollout perturbations to mass & inertia ----
        float m  = this->params_.m;
        float Ix = this->params_.Ix;
        float Iy = this->params_.Iy;
        float Iz = this->params_.Iz;

        // Don't perturb the first rollout (the nominal trajectory)
        if (rollout > 0) {
            m  = m  + fabsf((*(pert->m_pert_d_  ))(rollout));
            Ix = fabsf(Ix + (*(pert->ixx_pert_d_))(rollout));
            Iy = fabsf(Iy + (*(pert->iyy_pert_d_))(rollout));
            Iz = fabsf(Iz + (*(pert->izz_pert_d_))(rollout));
        }

        // ---- Unpack state ----
        float vx    = x(3);
        float vy    = x(4);
        float vz    = x(5);
        float phi   = x(6);   // roll
        float theta = x(7);   // pitch
        float psi   = x(8);   // yaw
        float p     = x(9);   // body roll  rate
        float q     = x(10);  // body pitch rate
        float r     = x(11);  // body yaw   rate

        // ---- Unpack control ----
        float U         = u(0);
        float tau_phi   = u(1);
        float tau_theta = u(2);
        float tau_psi   = u(3);

        // ---- Translational accelerations ----
        // Same body->inertial rotation as the cpp file, with names remapped
        // to standard convention.
        float sphi = sinf(phi),   cphi = cosf(phi);
        float sth  = sinf(theta), cth  = cosf(theta);
        float spsi = sinf(psi),   cpsi = cosf(psi);

        float ax =  (spsi*sphi + cpsi*sth*cphi) * U / m;
        float ay = (-cpsi*sphi + spsi*sth*cphi) * U / m;
        float az =   cth*cphi * U / m  -  this->params_.g;

        // ---- Angular accelerations (Euler's equations, simplified
        //      diagonal-inertia form, exactly as in the cpp code with names
        //      corrected to standard) ----
        float p_dot = ((Ix + Iz - Iy) * r * q + tau_phi)   / Ix;
        float q_dot = ((Iz - Ix - Iy) * p * r + tau_theta) / Iy;
        float r_dot = ((Ix + Iy - Iz) * p * q + tau_psi)   / Iz;

        // ---- Euler-angle rates ----
        // The cpp code integrates body rates directly into Euler angles
        // (small-angle approximation). We faithfully reproduce that:
        float phi_dot   = p;
        float theta_dot = q;
        float psi_dot   = r;

        // ---- Pack xdot ----
        xdot(0)  = vx;
        xdot(1)  = vy;
        xdot(2)  = vz;
        xdot(3)  = ax;
        xdot(4)  = ay;
        xdot(5)  = az;
        xdot(6)  = phi_dot;
        xdot(7)  = theta_dot;
        xdot(8)  = psi_dot;
        xdot(9)  = p_dot;
        xdot(10) = q_dot;
        xdot(11) = r_dot;
    }

    /*
     * GATE-required entry point. Called by integrators_eigen::rk4.
     */
    template<class PER_T>
    __host__ __device__ void computeStateDeriv(PER_T* pert, int timestep, int rollout,
                                                state_array& x_k,
                                                control_array& u,
                                                state_array& xdot)
    {
        quad_dynamics(pert, rollout, x_k, u, xdot);
    }
};

#endif // QUAD_DYNAMICS_CUH_
