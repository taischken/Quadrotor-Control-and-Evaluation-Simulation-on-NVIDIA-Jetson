#ifndef MISSION_REFERENCE_CUH_
#define MISSION_REFERENCE_CUH_

/*
 * SHARED mission definition and smooth reference-trajectory math.
 *
 * Used by:
 *   - GPU controller (quad_PID.cuh)        for setpoint generation
 *   - GPU scoring kernel                    for error measurement
 *   - CPU simulator                          for setpoint generation
 *   - Viewer                                  for drawing the reference path
 *
 * THE SINGLE SOURCE OF TRUTH for "where should the drone be at time t".
 * Every caller must use these functions; never re-implement the reference
 * math anywhere else.
 *
 * Mission structure (4 phases):
 *   Phase 1: climb     from (0,0,0) to (0,0,z1)         duration: phase_dur_1
 *   Phase 2: translate from (0,0,z1) to (x,y,z1)        duration: phase_dur_2
 *   Phase 3: altitude  from (x,y,z1) to (x,y,z2)        duration: phase_dur_3
 *   Phase 4: settle    hold at (x,y,z2)                 until end of sim
 *
 * Within each phase, the reference moves smoothly using a smoothstep curve
 * (s(p) = 3p^2 - 2p^3). This curve has zero derivative at p=0 and p=1, so
 * the reference starts and ends with zero velocity. This is much easier for
 * the PID to track than the original step-jumps.
 */

#include <cmath>

#ifdef __CUDACC__
#define MR_DEVICE __host__ __device__
#else
#define MR_DEVICE
#endif

struct MissionRef {
    // Waypoints (user-supplied)
    float z1       = 2.0f;       // first hover altitude
    float x_target = 3.0f;       // lateral target x
    float y_target = 2.0f;       // lateral target y
    float z2       = 2.0f;       // final altitude

    // Phase boundaries (in timesteps). Set up by the host before launch.
    //   Phase 1 occupies steps [0,             step_end_1)
    //   Phase 2 occupies steps [step_end_1,    step_end_2)
    //   Phase 3 occupies steps [step_end_2,    step_end_3)
    //   Phase 4 occupies steps [step_end_3,    num_timesteps)
    int step_end_1 = 1000;        // 5 s at dt=0.005
    int step_end_2 = 5000;        // start: 5 s; end depends on x/y distance
    int step_end_3 = 6000;        // +5 s for altitude change

    // For convenience -- shared by CPU/GPU code
    float dt = 0.005f;

    // Desired yaw (held throughout)
    float des_psi = 0.0f;
};

/*
 * Smoothstep: s(p) = 3p^2 - 2p^3 for p in [0,1], clamped at edges.
 * Has zero derivative at p=0 and p=1, so transitions are gentle.
 */
MR_DEVICE inline float smoothstep01(float p) {
    if (p <= 0.0f) return 0.0f;
    if (p >= 1.0f) return 1.0f;
    return p * p * (3.0f - 2.0f * p);
}

/*
 * Linear interpolation
 */
MR_DEVICE inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

/*
 * Compute the time-varying reference (des_x, des_y, des_z) at the given
 * simulation timestep. THE reference function for the whole system.
 */
MR_DEVICE inline void compute_reference(int timestep, const MissionRef& m,
                                         float& des_x, float& des_y, float& des_z)
{
    if (timestep < m.step_end_1) {
        // ---- Phase 1: climb from (0,0,0) to (0,0,z1) ----
        float p = (float)timestep / (float)(m.step_end_1 > 0 ? m.step_end_1 : 1);
        float s = smoothstep01(p);
        des_x = 0.0f;
        des_y = 0.0f;
        des_z = lerp(0.0f, m.z1, s);
    }
    else if (timestep < m.step_end_2) {
        // ---- Phase 2: translate from (0,0,z1) to (x,y,z1) ----
        int span = m.step_end_2 - m.step_end_1;
        float p = (float)(timestep - m.step_end_1) / (float)(span > 0 ? span : 1);
        float s = smoothstep01(p);
        des_x = lerp(0.0f,        m.x_target, s);
        des_y = lerp(0.0f,        m.y_target, s);
        des_z = m.z1;
    }
    else if (timestep < m.step_end_3) {
        // ---- Phase 3: altitude change from z1 to z2 ----
        int span = m.step_end_3 - m.step_end_2;
        float p = (float)(timestep - m.step_end_2) / (float)(span > 0 ? span : 1);
        float s = smoothstep01(p);
        des_x = m.x_target;
        des_y = m.y_target;
        des_z = lerp(m.z1, m.z2, s);
    }
    else {
        // ---- Phase 4: hold at final waypoint ----
        des_x = m.x_target;
        des_y = m.y_target;
        des_z = m.z2;
    }
}

/*
 * Host-side helper: compute phase durations for a given mission.
 * Translates "user's target" into "step_end_1, step_end_2, step_end_3".
 *
 *   - Phase 1 fixed at 5 s
 *   - Phase 2 scales with horizontal distance, at ~5 m/s cruise + 3 s overhead,
 *     bounded between 5 s and (num_timesteps*dt - 10) s
 *   - Phase 3 fixed at 5 s
 *   - Phase 4 (settling) consumes the remainder
 *
 * Call this on the host before launching, then set m.step_end_* on the
 * MissionRef that goes to the GPU.
 */
inline void compute_phase_boundaries(MissionRef& m, int num_timesteps,
                                      float cruise_speed = 5.0f)
{
    const float phase1_seconds = 5.0f;
    const float phase3_seconds = 5.0f;

    float horiz_dist = sqrtf(m.x_target * m.x_target + m.y_target * m.y_target);
    float phase2_seconds = horiz_dist / cruise_speed + 3.0f;

    // Clamp so phase 4 (settling) has at least 5 s
    float total_s = num_timesteps * m.dt;
    float reserved_s = phase1_seconds + phase3_seconds + 5.0f;   // +5s for settle
    if (phase2_seconds > total_s - reserved_s) {
        phase2_seconds = total_s - reserved_s;
    }
    if (phase2_seconds < 5.0f) phase2_seconds = 5.0f;

    m.step_end_1 = (int)(phase1_seconds / m.dt);
    m.step_end_2 = m.step_end_1 + (int)(phase2_seconds / m.dt);
    m.step_end_3 = m.step_end_2 + (int)(phase3_seconds / m.dt);
}

#endif // MISSION_REFERENCE_CUH_
