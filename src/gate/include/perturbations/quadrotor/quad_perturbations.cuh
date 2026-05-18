#ifndef QUAD_PERTURBATIONS_CUH_
#define QUAD_PERTURBATIONS_CUH_

#include <perturbations/perturbations_stream_managed.cuh>
#include <dynamics/quadrotor/quad_dynamics.cuh>
#include <cuda_util/cuda_memory_utils.cuh>

/*
 * Perturbations for the cpp-style quadrotor.
 *   STATE_DIM   = 12  (px,py,pz, vx,vy,vz, phi,theta,psi, p,q,r)
 *   CONTROL_DIM = 4   (U, tau_phi, tau_theta, tau_psi)
 *
 * Per-rollout perturbations:
 *   x0_pert_d_   : initial-state noise          [12 x NUM_ROLLOUTS]
 *   u_t_pert_d_  : control-input noise          [4 x NUM_ROLLOUTS*NUM_TIMESTEPS]
 *   m_pert_d_    : scalar mass perturbation     [NUM_ROLLOUTS]
 *   ixx,iyy,izz  : scalar inertia perturbations [NUM_ROLLOUTS]
 */

namespace quad_pert {
    const int S_DIM = 12;
    const int C_DIM = 4;
}

class QuadPertParams : public GATE_internal::PerturbationParam<quad_pert::S_DIM, quad_pert::C_DIM> {
public:
    QuadPertParams() : GATE_internal::PerturbationParam<quad_pert::S_DIM, quad_pert::C_DIM>() {}
    QuadPertParams(const state_array& x0_mean, const state_array& x0_std, const control_array& u_std) :
        GATE_internal::PerturbationParam<quad_pert::S_DIM, quad_pert::C_DIM>(x0_mean, x0_std, u_std) {}
    ~QuadPertParams() = default;

    // Standard deviations for parameter (mass / inertia) perturbations
    float m_std_   = 0.01f;
    float ixx_std_ = 0.0005f;   // smaller than the LQR version: keeps Ix>0 reliably
    float iyy_std_ = 0.0005f;
    float izz_std_ = 0.0005f;
};

template<int N_TIMESTEPS, int N_ROLLOUTS>
class QuadPert : public GATE_internal::Perturbations<QuadPert<N_TIMESTEPS, N_ROLLOUTS>,
    QuadPertParams, quad_pert::S_DIM, quad_pert::C_DIM, N_TIMESTEPS, N_ROLLOUTS> {
public:
    using Base = GATE_internal::Perturbations<QuadPert, QuadPertParams,
                                              quad_pert::S_DIM, quad_pert::C_DIM,
                                              N_TIMESTEPS, N_ROLLOUTS>;
    using scalar_all_rollouts = typename Base::scalar_all_rollouts;

    QuadPert(cudaStream_t stream = nullptr) : Base(stream) {
        QuadPert::allocateCUDAMem();
    }
    QuadPert(const QuadPertParams& params, cudaStream_t stream = nullptr) : Base(params, stream) {
        QuadPert::allocateCUDAMem();
    }
    ~QuadPert() {
        QuadPert::deallocateCudaMem();
    }

    // Device pointers
    scalar_all_rollouts* m_pert_d_   = nullptr;
    scalar_all_rollouts* ixx_pert_d_ = nullptr;
    scalar_all_rollouts* iyy_pert_d_ = nullptr;
    scalar_all_rollouts* izz_pert_d_ = nullptr;

    void initPerturbations() {
        scalar_all_rollouts m_pert_host   = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS,
            [&]() { return this->normal_distribution_(this->generator_); });
        scalar_all_rollouts ixx_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS,
            [&]() { return this->normal_distribution_(this->generator_); });
        scalar_all_rollouts iyy_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS,
            [&]() { return this->normal_distribution_(this->generator_); });
        scalar_all_rollouts izz_pert_host = scalar_all_rollouts::NullaryExpr(1, N_ROLLOUTS,
            [&]() { return this->normal_distribution_(this->generator_); });

        m_pert_host   = (m_pert_host   * this->params_.m_std_);
        ixx_pert_host = (ixx_pert_host * this->params_.ixx_std_);
        iyy_pert_host = (iyy_pert_host * this->params_.iyy_std_);
        izz_pert_host = (izz_pert_host * this->params_.izz_std_);

        HANDLE_ERROR(cudaMemcpyAsync(m_pert_d_,   &m_pert_host,
            sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, this->stream_));
        HANDLE_ERROR(cudaMemcpyAsync(ixx_pert_d_, &ixx_pert_host,
            sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, this->stream_));
        HANDLE_ERROR(cudaMemcpyAsync(iyy_pert_d_, &iyy_pert_host,
            sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, this->stream_));
        HANDLE_ERROR(cudaMemcpyAsync(izz_pert_d_, &izz_pert_host,
            sizeof(scalar_all_rollouts), cudaMemcpyHostToDevice, this->stream_));

        CudaCheckError();
    }

private:
    void allocateCUDAMem() {
        GATE_internal::cudaObjectMalloc(m_pert_d_);
        GATE_internal::cudaObjectMalloc(ixx_pert_d_);
        GATE_internal::cudaObjectMalloc(iyy_pert_d_);
        GATE_internal::cudaObjectMalloc(izz_pert_d_);
    }

    void deallocateCudaMem() {
        GATE_internal::cudaObjectFree(m_pert_d_);
        GATE_internal::cudaObjectFree(ixx_pert_d_);
        GATE_internal::cudaObjectFree(iyy_pert_d_);
        GATE_internal::cudaObjectFree(izz_pert_d_);
    }
};

#endif // QUAD_PERTURBATIONS_CUH_
