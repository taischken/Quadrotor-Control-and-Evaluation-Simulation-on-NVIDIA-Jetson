# Real-Time Drone Application — Setup Guide

This guide explains how to add GLFW and OpenGL to your existing CUDA Visual Studio project so the new application files compile and run.

## Files in this package

| File | Where it goes in your project |
|---|---|
| `kernel.cu` | Replace your existing `kernel.cu` |
| `cpu_simulator.h` | Add to project root (alongside `kernel.cu`) |
| `viewer.h` | Add to project root |
| `viewer.cpp` | Add to project root |

## Library dependencies

Two libraries beyond what you already have:

1. **GLFW** — windowing, mouse/keyboard input
2. **OpenGL** — already present on every Windows system, no installation needed

GLM (matrix math library) is **not** needed — the viewer uses manually computed transformation matrices to avoid the extra dependency.

---

## Step 1: Download GLFW

1. Go to https://www.glfw.org/download.html
2. Download **64-bit Windows binaries** (a zip file like `glfw-3.4.bin.WIN64.zip`)
3. Extract to a stable location, e.g. `C:\libs\glfw-3.4.bin.WIN64\`

Inside that folder you'll find:
- `include\GLFW\glfw3.h` — the header
- `lib-vc2022\glfw3.lib` (or `lib-vc2019\` if you use VS 2019) — the static library
- `lib-vc2022\glfw3.dll` — runtime DLL (we won't use this; we'll link statically)

For Visual Studio 2026 / 2022, use `lib-vc2022`. For older versions, use the matching folder.

---

## Step 2: Configure your Visual Studio project

Right-click your project in Solution Explorer → **Properties**.

Set the **Configuration** dropdown to **All Configurations** (or to the specific one you're building) so the settings apply everywhere.

### 2a. Add the include path

Navigate to **C/C++ → General → Additional Include Directories**.

Click the dropdown arrow → **<Edit...>** → click the "New Line" icon → add:

```
C:\libs\glfw-3.4.bin.WIN64\include
```

### 2b. Add the library path

Navigate to **Linker → General → Additional Library Directories**.

Edit and add:

```
C:\libs\glfw-3.4.bin.WIN64\lib-vc2022
```

### 2c. Add the libraries to link

Navigate to **Linker → Input → Additional Dependencies**.

Edit and add (each on its own line):

```
glfw3.lib
opengl32.lib
gdi32.lib
user32.lib
shell32.lib
```

These last four are Windows system libraries that GLFW needs. They're always available on Windows; you just have to tell the linker to use them.

### 2d. Disable static-runtime mismatch warnings (if you hit them)

If you get linker errors like `LIBCMT.lib vs MSVCRT.lib mismatch`, navigate to:

**C/C++ → Code Generation → Runtime Library**

Set it to **Multi-threaded DLL (/MD)** for Release, or **Multi-threaded Debug DLL (/MDd)** for Debug. This matches GLFW's default build.

---

## Step 3: Add the new files to the project

In Solution Explorer, right-click your project → **Add → Existing Item...** and select:

- `kernel.cu` (replaces your existing one — delete the old one first or VS will refuse)
- `cpu_simulator.h`
- `viewer.h`
- `viewer.cpp`

Make sure `viewer.cpp` is treated as **C++**, not CUDA. Right-click `viewer.cpp` → Properties → **General → Item Type** → ensure it's **C/C++ compiler**, not **CUDA C/C++**. (Files in your project list that have no `.cu` extension default to C++, but it's worth verifying.)

`kernel.cu` stays as CUDA C/C++.

---

## Step 4: Build

**Clean + Rebuild** the solution. Header-only changes don't always trigger a rebuild correctly, so a clean rebuild is safer.

Common errors and fixes:

| Error | Fix |
|---|---|
| `cannot open include file: 'GLFW/glfw3.h'` | Step 2a path is wrong |
| `unresolved external symbol __imp_glfwInit` | Step 2b or 2c is wrong |
| `LIBCMT.lib(...) already defined in MSVCRT.lib` | Step 2d — set runtime library to /MD |
| `cannot open file 'glfw3.lib'` | Wrong subfolder; check `lib-vc2022` vs `lib-vc2019` |
| Linker error about `opengl32.lib` | Add `opengl32.lib` in Step 2c |

---

## Step 5: Run

When you run the program, a console window appears first:

```
===================================================
       Real-Time Drone Mission Configuration       
===================================================

Enter z1 (first hover altitude, meters)        : 2
Enter target x (meters)                        : 3
Enter target y (meters)                        : 2
Enter z2 (final altitude after x-y move, meters): 4
```

Type your mission, press Enter. The program will:

1. Print "Running GPU rollout kernel..." (takes ~0.5 s)
2. Print "Running GPU scoring kernel..." (takes ~0.05 s)
3. Show the winning rollout's gains
4. **Open the OpenGL window** with the animation

### In the viewer

- **Drag with the mouse**: rotate the camera around the scene
- **Scroll wheel**: zoom in/out
- **SPACE**: pause/resume the animation
- **R**: reset the animation back to t=0 (note: only resets the visualization; the CPU simulator state isn't reset — see "limitations" below)
- **ESC** or click the X: close the window

You should see **two drones** flying through the scene:
- **Blue drone** = GPU replay (the trajectory computed during the kernel run)
- **Orange drone** = live CPU simulation (running step-by-step with the winning gains)

They should track each other very closely. If they diverge significantly, something is wrong (different gains, different setpoints, or a numerical bug).

The grey dashed line is the reference path through the four waypoints (origin → top of z1 → (x,y,z1) → (x,y,z2)). Orange dots mark each waypoint.

---

## Limitations and known issues

1. **R (reset) only resets the trail visualization.** The CPU simulator's internal state (PID integrals, current step) isn't reset because `Simulator` doesn't have a reset method. To fully reset, close and reopen the program.

2. **Window title is the HUD.** There's no font system in fixed-function OpenGL. The current step number and pause state are shown in the window title bar. To add a proper text HUD, integrate Dear ImGui (separate effort).

3. **Mission Phase 3 (the z1→z2 change) is only simulated on the CPU side.** The GPU kernel still uses the 2-phase mission for its search (climb to z1, then move to xy_target). The CPU simulator uses 3 phases, including the final altitude change to z2. So during phase 3, only the orange drone will move — the blue one will already be done.

4. **40 seconds in real time is long.** The animation plays at 1× speed by default, meaning a 40-second simulation takes 40 seconds to watch. Change `v.playback_speed = 2.0f` (or higher) in `kernel.cu` to speed it up.

5. **All drone positions are in world frame.** The visualization assumes the same convention as the simulation: X = North, Y = East, Z = Up.

---

## Project file checklist

Your `Solution Explorer` should look like this after setup:

```
YourSolution
└─ YourProject
   ├─ kernel.cu              (modified — main entry point)
   ├─ cpu_simulator.h        (new)
   ├─ viewer.h               (new)
   ├─ viewer.cpp             (new)
   └─ gate/include/...        (your existing GATE headers, unchanged)
```

With include directories:
```
$(ProjectDir)
$(ProjectDir)\gate\include
$(ProjectDir)\eigen-3.3.8
C:\libs\glfw-3.4.bin.WIN64\include
```

If everything is wired correctly, the build should succeed and you'll be flying drones in real time within minutes.
