/*
 * viewer.cpp -- 3D drone visualization.
 *
 * Now draws the SMOOTH reference path (computed via shared mission_reference)
 * instead of step-jumps between waypoints. The reference line is the actual
 * curve the controller is trying to follow.
 */

#include "viewer.h"

#include <GLFW/glfw3.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <chrono>
#include <vector>
#include <thread>

// -----------------------------------------------------------------------------
// Camera
// -----------------------------------------------------------------------------
struct Camera {
    float distance = 30.0f;       // a bit larger default for long missions
    float yaw      = 0.6f;
    float pitch    = 0.4f;
    float target_x = 0.0f;
    float target_y = 0.0f;
    float target_z = 5.0f;
};

static Camera g_cam;
static bool   g_mouse_dragging = false;
static double g_last_mouse_x = 0, g_last_mouse_y = 0;
static bool   g_paused = false;
static bool   g_reset_requested = false;

static void key_callback(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE)  g_paused = !g_paused;
    if (key == GLFW_KEY_R)      g_reset_requested = true;
}

static void mouse_button_callback(GLFWwindow* w, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse_dragging = (action == GLFW_PRESS);
        glfwGetCursorPos(w, &g_last_mouse_x, &g_last_mouse_y);
    }
}

static void cursor_pos_callback(GLFWwindow*, double x, double y) {
    if (g_mouse_dragging) {
        double dx = x - g_last_mouse_x;
        double dy = y - g_last_mouse_y;
        g_cam.yaw   += (float)(dx * 0.01);
        g_cam.pitch += (float)(dy * 0.01);
        if (g_cam.pitch >  1.5f) g_cam.pitch =  1.5f;
        if (g_cam.pitch < -0.2f) g_cam.pitch = -0.2f;
    }
    g_last_mouse_x = x;
    g_last_mouse_y = y;
}

static void scroll_callback(GLFWwindow*, double, double yoff) {
    g_cam.distance *= (yoff > 0) ? 0.9f : 1.1f;
    if (g_cam.distance < 2.0f)   g_cam.distance = 2.0f;
    if (g_cam.distance > 500.0f) g_cam.distance = 500.0f;
}

// -----------------------------------------------------------------------------
// Auto-frame the camera to fit the mission
// -----------------------------------------------------------------------------
static void auto_frame_camera(const MissionRef& m) {
    g_cam.target_x = m.x_target * 0.5f;
    g_cam.target_y = m.y_target * 0.5f;
    g_cam.target_z = (m.z1 + m.z2) * 0.5f;

    float horiz = sqrtf(m.x_target*m.x_target + m.y_target*m.y_target);
    float vert  = fmaxf(m.z1, m.z2);
    float extent = fmaxf(horiz, vert);
    if (extent < 5.0f) extent = 5.0f;
    g_cam.distance = extent * 2.5f;
}

// -----------------------------------------------------------------------------
// Camera transform
// -----------------------------------------------------------------------------
static void apply_camera() {
    float eye_x = g_cam.target_x + g_cam.distance * cosf(g_cam.pitch) * sinf(g_cam.yaw);
    float eye_y = g_cam.target_y + g_cam.distance * cosf(g_cam.pitch) * cosf(g_cam.yaw);
    float eye_z = g_cam.target_z + g_cam.distance * sinf(g_cam.pitch);

    float fx = g_cam.target_x - eye_x;
    float fy = g_cam.target_y - eye_y;
    float fz = g_cam.target_z - eye_z;
    float fn = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= fn; fy /= fn; fz /= fn;

    float upx = 0, upy = 0, upz = 1;
    float sx = fy*upz - fz*upy;
    float sy = fz*upx - fx*upz;
    float sz = fx*upy - fy*upx;
    float sn = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= sn; sy /= sn; sz /= sn;

    float ux = sy*fz - sz*fy;
    float uy = sz*fx - sx*fz;
    float uz = sx*fy - sy*fx;

    float m[16] = {
        sx, ux, -fx, 0,
        sy, uy, -fy, 0,
        sz, uz, -fz, 0,
        0,   0,   0, 1
    };
    glMultMatrixf(m);
    glTranslatef(-eye_x, -eye_y, -eye_z);
}

// -----------------------------------------------------------------------------
// Scene primitives
// -----------------------------------------------------------------------------
static void draw_grid(float extent) {
    glColor3f(0.3f, 0.3f, 0.35f);
    glLineWidth(1.0f);
    // Cell size: 1 m up to extent=10, 5 m up to 50, 10 m beyond
    float cell = (extent < 10.0f) ? 1.0f : (extent < 50.0f ? 5.0f : 10.0f);
    int   half = (int)ceilf(extent / cell);
    glBegin(GL_LINES);
    for (int i = -half; i <= half; ++i) {
        glVertex3f(i*cell, -half*cell, 0); glVertex3f(i*cell, half*cell, 0);
        glVertex3f(-half*cell, i*cell, 0); glVertex3f(half*cell, i*cell, 0);
    }
    glEnd();
}

static void draw_axes(float len) {
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1, 0.2f, 0.2f); glVertex3f(0,0,0); glVertex3f(len, 0, 0);
    glColor3f(0.2f, 1, 0.2f); glVertex3f(0,0,0); glVertex3f(0, len, 0);
    glColor3f(0.2f, 0.5f, 1); glVertex3f(0,0,0); glVertex3f(0, 0, len);
    glEnd();
}

static void draw_waypoint_marker(float x, float y, float z) {
    glColor3f(1.0f, 0.7f, 0.0f);
    glPointSize(12.0f);
    glBegin(GL_POINTS); glVertex3f(x, y, z); glEnd();

    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glVertex3f(x, y, 0); glVertex3f(x, y, z);
    glEnd();
}

/*
 * Draw the SMOOTH reference path by sampling the shared compute_reference()
 * function at many timesteps. This is what the drone actually tracks.
 */
static void draw_reference_path(const MissionRef& mission, int total_steps) {
    const int N = 400;   // sample density along the path
    glColor3f(0.6f, 0.6f, 0.6f);
    glLineWidth(2.0f);
    glLineStipple(2, 0xAAAA);
    glEnable(GL_LINE_STIPPLE);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < N; ++i) {
        int step = (int)((float)i / (N - 1) * total_steps);
        float rx, ry, rz;
        compute_reference(step, mission, rx, ry, rz);
        glVertex3f(rx, ry, rz);
    }
    glEnd();
    glDisable(GL_LINE_STIPPLE);

    // Waypoint markers at start of each phase
    draw_waypoint_marker(0, 0, 0);
    draw_waypoint_marker(0, 0, mission.z1);
    draw_waypoint_marker(mission.x_target, mission.y_target, mission.z1);
    draw_waypoint_marker(mission.x_target, mission.y_target, mission.z2);
}

static void draw_drone_at(float x, float y, float z,
                           float phi, float theta, float psi,
                           float r, float g, float b)
{
    glPushMatrix();
    glTranslatef(x, y, z);
    glRotatef(psi * 180.0f / (float)M_PI,   0, 0, 1);
    glRotatef(theta * 180.0f / (float)M_PI, 0, 1, 0);
    glRotatef(phi * 180.0f / (float)M_PI,   1, 0, 0);

    const float arm = 0.25f;
    glColor3f(r, g, b);
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glVertex3f(-arm, 0, 0); glVertex3f(arm, 0, 0);
    glVertex3f(0, -arm, 0); glVertex3f(0, arm, 0);
    glEnd();

    glColor3f(r*0.7f + 0.3f, g*0.7f + 0.3f, b*0.7f + 0.3f);
    glLineWidth(1.5f);
    const int N = 16;
    const float rad = 0.10f;
    for (int side = 0; side < 4; ++side) {
        float cx = 0, cy = 0;
        switch (side) {
            case 0: cx =  arm; break;
            case 1: cx = -arm; break;
            case 2: cy =  arm; break;
            case 3: cy = -arm; break;
        }
        glBegin(GL_LINE_LOOP);
        for (int k = 0; k < N; ++k) {
            float a = 2.0f * (float)M_PI * k / N;
            glVertex3f(cx + rad*cosf(a), cy + rad*sinf(a), 0);
        }
        glEnd();
    }
    glColor3f(1, 1, 1);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0); glVertex3f(arm*1.4f, 0, 0);
    glEnd();
    glPopMatrix();
}

static void draw_trail(const std::vector<float>& xs,
                       const std::vector<float>& ys,
                       const std::vector<float>& zs,
                       float r, float g, float b)
{
    glColor3f(r, g, b);
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < xs.size(); ++i)
        glVertex3f(xs[i], ys[i], zs[i]);
    glEnd();
}

static void update_window_title(GLFWwindow* w, int step, int total_steps,
                                  int best_rollout, bool paused, float playback_x)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
        "Drone Real-Time Sim | step %d/%d | best rollout: %d | %s | %.1fx",
        step, total_steps, best_rollout, paused ? "PAUSED" : "playing", playback_x);
    glfwSetWindowTitle(w, buf);
}

// -----------------------------------------------------------------------------
// Main viewer function
// -----------------------------------------------------------------------------
void run_viewer(ViewerData& data) {
    if (!glfwInit()) {
        fprintf(stderr, "glfwInit failed\n");
        return;
    }

    GLFWwindow* window = glfwCreateWindow(1280, 800,
        "Drone Real-Time Simulation", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.08f, 0.09f, 0.12f, 1.0f);

    // Auto-frame the camera for the mission scale
    auto_frame_camera(data.mission);

    using Clock = std::chrono::steady_clock;
    auto start_time = Clock::now();

    std::vector<float> gpu_trail_x, gpu_trail_y, gpu_trail_z;
    std::vector<float> cpu_trail_x, cpu_trail_y, cpu_trail_z;

    int total_steps = data.gpu_num_steps;

    // Compute grid extent for nice scale
    float horiz_extent = fmaxf(fabsf(data.mission.x_target),
                                fabsf(data.mission.y_target));
    horiz_extent = fmaxf(horiz_extent, fmaxf(data.mission.z1, data.mission.z2));
    horiz_extent = fmaxf(horiz_extent, 5.0f);
    float axes_len = horiz_extent * 0.15f;

    while (!glfwWindowShouldClose(window)) {

        if (g_reset_requested) {
            g_reset_requested = false;
            start_time = Clock::now();
            gpu_trail_x.clear(); gpu_trail_y.clear(); gpu_trail_z.clear();
            cpu_trail_x.clear(); cpu_trail_y.clear(); cpu_trail_z.clear();
        }

        auto now = Clock::now();
        double wall_seconds = std::chrono::duration<double>(now - start_time).count();
        int sim_step;
        if (g_paused) {
            sim_step = (int)(wall_seconds / data.dt * data.playback_speed);
            start_time = now - std::chrono::milliseconds(
                (long long)(sim_step * data.dt * 1000.0 / data.playback_speed));
        } else {
            sim_step = (int)(wall_seconds * data.playback_speed / data.dt);
        }

        if (sim_step >= total_steps) sim_step = total_steps - 1;

        // GPU drone state at current step
        float gpu_x[12] = {0};
        if (data.gpu_trajectory && sim_step >= 0 && sim_step < total_steps) {
            for (int i = 0; i < 12; ++i)
                gpu_x[i] = data.gpu_trajectory[sim_step * 12 + i];
        }

        // Step CPU sim to current step
        if (!g_paused && data.cpu_simulator) {
            while (data.cpu_simulator->current_step() < sim_step
                   && !data.cpu_simulator->done()) {
                data.cpu_simulator->step();
            }
        }
        const float* cpu_x = data.cpu_simulator ? data.cpu_simulator->state() : gpu_x;

        if (sim_step % 4 == 0) {
            gpu_trail_x.push_back(gpu_x[0]);
            gpu_trail_y.push_back(gpu_x[1]);
            gpu_trail_z.push_back(gpu_x[2]);
            cpu_trail_x.push_back(cpu_x[0]);
            cpu_trail_y.push_back(cpu_x[1]);
            cpu_trail_z.push_back(cpu_x[2]);
        }

        // Render
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = (float)fb_w / (float)fb_h;
        // Far plane scales with mission extent
        float far_plane = horiz_extent * 10.0f + 200.0f;
        float n = 0.5f;
        float t = n * tanf(0.5f * (float)M_PI/3.0f);
        float rr = t * aspect;
        glFrustum(-rr, rr, -t, t, n, far_plane);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        apply_camera();

        draw_grid(horiz_extent * 1.2f);
        draw_axes(axes_len);
        draw_reference_path(data.mission, total_steps);

        draw_trail(gpu_trail_x, gpu_trail_y, gpu_trail_z, 0.4f, 0.7f, 1.0f);
        draw_trail(cpu_trail_x, cpu_trail_y, cpu_trail_z, 1.0f, 0.5f, 0.2f);

        draw_drone_at(gpu_x[0], gpu_x[1], gpu_x[2],
                      gpu_x[6], gpu_x[7], gpu_x[8],
                      0.4f, 0.7f, 1.0f);
        draw_drone_at(cpu_x[0], cpu_x[1], cpu_x[2],
                      cpu_x[6], cpu_x[7], cpu_x[8],
                      1.0f, 0.5f, 0.2f);

        update_window_title(window, sim_step, total_steps,
                            data.best_rollout_idx, g_paused, data.playback_speed);

        glfwSwapBuffers(window);
        glfwPollEvents();

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
