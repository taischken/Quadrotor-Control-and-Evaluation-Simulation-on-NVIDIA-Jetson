#ifndef SCORING_HOST_H_
#define SCORING_HOST_H_

/*
 * Host-side driver for the scoring kernel. Provides:
 *   - run_scoring_kernel(...)   : launch scoring kernel, copy results back
 *   - top_k(...)                : pure-host top-K of a cost array
 *   - print_top_k_table(...)    : pretty-print results with gain values
 *   - save_scores_csv(...)      : append a 'scores' CSV alongside trajectories
 */

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

#include <scoring/scoring_kernel.cuh>
#include <controllers/quadrotor/quad_PID.cuh>   // PIDGainBlock

namespace scoring {

    /*
     * Bundle of all per-rollout costs computed on the GPU.
     * After run_scoring_kernel(), these vectors live on the host
     * with one entry per rollout.
     */
    struct ScoreResults {
        std::vector<float> rmse;
        std::vector<float> settle;
        std::vector<float> overshoot;
        std::vector<float> combined;
        std::vector<int>   crashed;
    };

    /*
     * Allocate device buffers, launch scoring kernel, copy back to host.
     *
     * INPUTS:
     *   d_trajectories : device pointer to the [R][T][S] trajectory buffer
     *                    (already produced by computeTrajectories())
     *   num_rollouts, num_timesteps, state_dim : geometry
     *   dt             : sim timestep (used to convert step counts to seconds)
     *   sp             : scoring parameters (setpoints, weights, tolerances)
     *
     * OUTPUT: ScoreResults populated with one entry per rollout.
     */
    inline ScoreResults run_scoring_kernel(
        const float* d_trajectories,
        int num_rollouts,
        int num_timesteps,
        int state_dim,
        float dt,
        const ScoringParams& sp = ScoringParams{})
    {
        // ---- Allocate device output buffers (small: num_rollouts * 4 bytes each) ----
        float *d_rmse, *d_settle, *d_overshoot, *d_combined;
        int   *d_crashed;
        cudaMalloc(&d_rmse,      sizeof(float) * num_rollouts);
        cudaMalloc(&d_settle,    sizeof(float) * num_rollouts);
        cudaMalloc(&d_overshoot, sizeof(float) * num_rollouts);
        cudaMalloc(&d_combined,  sizeof(float) * num_rollouts);
        cudaMalloc(&d_crashed,   sizeof(int)   * num_rollouts);

        // ---- Launch ----
        const int bdim = 128;                                 // 4 warps per block
        const int grid = (num_rollouts + bdim - 1) / bdim;    // ceil-divide
        scoreRolloutsKernel<<<grid, bdim>>>(
            d_trajectories, num_rollouts, num_timesteps, state_dim,
            dt, sp,
            d_rmse, d_settle, d_overshoot, d_combined, d_crashed);

        // Force errors (if any) to surface here, not later
        cudaDeviceSynchronize();

        // ---- Copy results to host ----
        ScoreResults out;
        out.rmse     .resize(num_rollouts);
        out.settle   .resize(num_rollouts);
        out.overshoot.resize(num_rollouts);
        out.combined .resize(num_rollouts);
        out.crashed  .resize(num_rollouts);

        cudaMemcpy(out.rmse.data(),      d_rmse,      sizeof(float)*num_rollouts, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.settle.data(),    d_settle,    sizeof(float)*num_rollouts, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.overshoot.data(), d_overshoot, sizeof(float)*num_rollouts, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.combined.data(),  d_combined,  sizeof(float)*num_rollouts, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.crashed.data(),   d_crashed,   sizeof(int)  *num_rollouts, cudaMemcpyDeviceToHost);

        // ---- Free device buffers ----
        cudaFree(d_rmse); cudaFree(d_settle); cudaFree(d_overshoot);
        cudaFree(d_combined); cudaFree(d_crashed);

        return out;
    }

    /*
     * Return the indices of the K smallest values in `costs`, ascending.
     * Uses partial_sort -- O(N log K), perfectly fine for N=1024.
     *
     * If `exclude_crashed` is true, indices with crashed[i]==1 are skipped.
     */
    inline std::vector<int> top_k(const std::vector<float>& costs,
                                   int k,
                                   const std::vector<int>* crashed = nullptr,
                                   bool exclude_crashed = true)
    {
        std::vector<int> idx;
        idx.reserve(costs.size());
        for (int i = 0; i < (int)costs.size(); ++i) {
            if (exclude_crashed && crashed && (*crashed)[i]) continue;
            idx.push_back(i);
        }
        if ((int)idx.size() <= k) {
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return costs[a] < costs[b]; });
            return idx;
        }
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return costs[a] < costs[b]; });
        idx.resize(k);
        return idx;
    }

    /*
     * Pretty-print one top-K list with the rollout's gain values.
     */
    inline void print_top_k_table(const std::string& metric_name,
                                   const std::vector<int>& top_idx,
                                   const std::vector<float>& costs,
                                   const std::vector<PIDGainBlock>& gains)
    {
        std::cout << "\n=== TOP " << top_idx.size() << " by " << metric_name << " ===\n";
        std::cout << std::left
                  << std::setw(5)  << "rank"
                  << std::setw(8)  << "rollout"
                  << std::setw(12) << "cost"
                  << std::setw(10) << "kp_z"
                  << std::setw(10) << "kd_z"
                  << std::setw(10) << "kp_phi"
                  << std::setw(10) << "kd_phi"
                  << std::setw(10) << "kp_theta"
                  << std::setw(10) << "kd_theta"
                  << "\n";
        std::cout << std::string(85, '-') << "\n";
        for (int rank = 0; rank < (int)top_idx.size(); ++rank) {
            int r = top_idx[rank];
            const PIDGainBlock& g = gains[r];
            std::cout << std::left << std::fixed << std::setprecision(3)
                      << std::setw(5)  << (rank + 1)
                      << std::setw(8)  << r
                      << std::setw(12) << costs[r]
                      << std::setw(10) << g.kp[0]
                      << std::setw(10) << g.kd[0]
                      << std::setw(10) << g.kp[3]
                      << std::setw(10) << g.kd[3]
                      << std::setw(10) << g.kp[4]
                      << std::setw(10) << g.kd[4]
                      << "\n";
        }
    }

    /*
     * Save all per-rollout costs to a CSV next to the trajectory file.
     * Same datetime stem as the trajectories+gains, so the three files
     * share a common base name.
     */
    inline void save_scores_csv(const std::string& trajectory_dir,
                                 const std::string& system_name,
                                 const std::string& datetime_stem,
                                 const ScoreResults& s)
    {
        std::ofstream f(trajectory_dir + system_name + datetime_stem + "_scores.csv");
        f << "rollout,crashed,rmse,settle,overshoot,combined\n";
        for (int r = 0; r < (int)s.rmse.size(); ++r) {
            f << r << "," << s.crashed[r] << ","
              << s.rmse[r] << "," << s.settle[r] << ","
              << s.overshoot[r] << "," << s.combined[r] << "\n";
        }
        std::cout << "Saved scores to " << system_name + datetime_stem + "_scores.csv\n";
    }

}   // namespace scoring

#endif // SCORING_HOST_H_
