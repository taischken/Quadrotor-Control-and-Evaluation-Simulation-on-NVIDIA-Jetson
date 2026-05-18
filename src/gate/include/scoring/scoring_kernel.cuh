#ifndef SCORING_KERNEL_CUH_
#define SCORING_KERNEL_CUH_

/*
 * GPU SCORING for the quadrotor gain sweep -- now mission-aware.
 *
 * Uses the SAME smooth reference as the controller (via mission_reference.cuh),
 * so RMSE measures actual tracking error rather than error against an
 * unreachable step-target.
 *
 * Costs computed per rollout:
 *   cost_rmse[r]      : sqrt(mean over t of ||p(t) - p_ref(t)||^2)
 *   cost_settle[r]    : time after end-of-mission for the drone to enter
 *                       and stay within a tolerance band of the final waypoint
 *   cost_overshoot[r] : peak deviation past the reference (any axis)
 *                       sampled during phases 2-3
 *   cost_combined[r]  : weighted sum of the above + crash penalty
 *
 * Crash bounds are now generous (500 m) so long missions don't false-positive.
 */

#include <cuda_runtime.h>
#include <mission_reference.cuh>

namespace scoring {

    struct ScoringParams {
        MissionRef mission;          // <-- NEW: shared mission

        // Settling tolerance after the final waypoint is reached
        float settle_tol = 0.20f;    // 20 cm (was 10 cm)

        // Crash detection -- generous, only catches true tumbling
        float crash_z_min   = -10.0f;     // dipped > 10 m below ground
        float crash_pos_max = 500.0f;     // any axis > 500 m
        float crash_ang_max = 5.0f;       // any euler angle > 5 rad

        // Cost weights
        float w_rmse      = 1.0f;
        float w_settle    = 0.05f;
        float w_overshoot = 0.5f;
        float w_crash     = 1000.0f;
    };

    __global__ void scoreRolloutsKernel(
        const float* __restrict__ trajectories,
        int   num_rollouts,
        int   num_timesteps,
        int   state_dim,
        float dt,
        ScoringParams sp,
        float* cost_rmse,
        float* cost_settle,
        float* cost_overshoot,
        float* cost_combined,
        int*   crashed_out)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= num_rollouts) return;

        const float* traj = trajectories + tid * num_timesteps * state_dim;

        // ------------- Pass 1: crash detection -------------
        int crashed = 0;
        for (int k = 0; k < num_timesteps; ++k) {
            float px = traj[k * state_dim + 0];
            float py = traj[k * state_dim + 1];
            float pz = traj[k * state_dim + 2];
            float phi   = traj[k * state_dim + 6];
            float theta = traj[k * state_dim + 7];
            float psi   = traj[k * state_dim + 8];

            float ap = fmaxf(fmaxf(fabsf(px), fabsf(py)), fabsf(pz));
            float aa = fmaxf(fmaxf(fabsf(phi), fabsf(theta)), fabsf(psi));

            if (pz < sp.crash_z_min)      crashed = 1;
            if (ap > sp.crash_pos_max)    crashed = 1;
            if (aa > sp.crash_ang_max)    crashed = 1;

            if (isnan(px) || isnan(py) || isnan(pz) ||
                isnan(phi) || isnan(theta) || isnan(psi))
                crashed = 1;
        }

        // ------------- Pass 2: cost accumulation -------------
        float sse = 0.0f;
        float overshoot_x = 0.0f, overshoot_y = 0.0f, overshoot_z = 0.0f;
        int   last_out_of_band = -1;

        for (int k = 0; k < num_timesteps; ++k) {
            float px = traj[k * state_dim + 0];
            float py = traj[k * state_dim + 1];
            float pz = traj[k * state_dim + 2];

            // ---- Compute the SAME smooth reference the controller saw ----
            float rx, ry, rz;
            compute_reference(k, sp.mission, rx, ry, rz);

            // RMSE -- accumulate squared error over all timesteps
            float ex = px - rx, ey = py - ry, ez = pz - rz;
            sse += ex*ex + ey*ey + ez*ez;

            // Overshoot -- measured during phases 2 and 3 (actual motion phases)
            if (k >= sp.mission.step_end_1 && k < sp.mission.step_end_3) {
                // For x and y, overshoot = how far past the target we went
                // in the direction of motion.
                float ox = (sp.mission.x_target > 0)
                            ? fmaxf(0.0f, px - sp.mission.x_target)
                            : fmaxf(0.0f, sp.mission.x_target - px);
                float oy = (sp.mission.y_target > 0)
                            ? fmaxf(0.0f, py - sp.mission.y_target)
                            : fmaxf(0.0f, sp.mission.y_target - py);
                float oz = fmaxf(0.0f, fabsf(pz - rz) - sp.settle_tol);

                if (ox > overshoot_x) overshoot_x = ox;
                if (oy > overshoot_y) overshoot_y = oy;
                if (oz > overshoot_z) overshoot_z = oz;
            }

            // Settling -- check distance from FINAL waypoint after phase 3 ends
            if (k >= sp.mission.step_end_3) {
                float dx = px - sp.mission.x_target;
                float dy = py - sp.mission.y_target;
                float dz = pz - sp.mission.z2;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist > sp.settle_tol) last_out_of_band = k;
            }
        }

        // ------------- Reduce to scalar costs -------------
        float total_steps = (float)num_timesteps;
        float rmse = sqrtf(sse / (total_steps * 3.0f));

        float settle_time;
        if (last_out_of_band < 0) {
            // Always within tolerance after phase 3 -> instant settle
            settle_time = 0.0f;
        } else if (last_out_of_band >= num_timesteps - 1) {
            // Never settled -> max penalty
            settle_time = (num_timesteps - sp.mission.step_end_3) * dt;
        } else {
            settle_time = (last_out_of_band + 1 - sp.mission.step_end_3) * dt;
            if (settle_time < 0) settle_time = 0.0f;
        }

        float overshoot_total = overshoot_x + overshoot_y + overshoot_z;

        float combined = sp.w_rmse      * rmse
                       + sp.w_settle    * settle_time
                       + sp.w_overshoot * overshoot_total
                       + sp.w_crash     * (float)crashed;

        cost_rmse[tid]      = rmse;
        cost_settle[tid]    = settle_time;
        cost_overshoot[tid] = overshoot_total;
        cost_combined[tid]  = combined;
        crashed_out[tid]    = crashed;
    }

}   // namespace scoring

#endif // SCORING_KERNEL_CUH_
