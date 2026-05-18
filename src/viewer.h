#ifndef VIEWER_H_
#define VIEWER_H_

/*
 * Real-time 3D viewer.
 *
 * Now uses the shared MissionRef to draw the SMOOTH reference path that
 * the controller actually tracks (instead of a step-function approximation).
 */

#include "cpu_simulator.h"
#include "mission_reference.cuh"

struct ViewerData {
    const float* gpu_trajectory = nullptr;
    int          gpu_num_steps  = 0;
    cpu_sim::Simulator* cpu_simulator = nullptr;
    MissionRef mission;
    float dt = 0.005f;
    int   best_rollout_idx = 0;
    float playback_speed = 1.0f;
};

void run_viewer(ViewerData& data);

#endif // VIEWER_H_
