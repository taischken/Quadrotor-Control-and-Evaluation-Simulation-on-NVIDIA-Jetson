#ifndef TRAJECTORY_EXPORT_H
#define TRAJECTORY_EXPORT_H

/* 
* These functions will be used to export trajectories in a more compact manner.
*/

#include <vector>
#include <string>
#include <direct.h>
#include <fstream>
#include <ctime>
#include <iostream>
#include <chrono>


namespace GATE_internal {

	std::string project_root = "C:/Users/berka/Desktop/Work/lw_gate_project";

	std::string create_save_location(std::string system_name, std::string project_root_path = project_root) {
		std::string trajectory_path = project_root_path + "/";
		return trajectory_path;
	}

	std::string get_datetime() {
		std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
		std::time_t start_time = std::chrono::system_clock::to_time_t(now);
		char timedisplay[100];
		tm buf;
		errno_t err = localtime_s(&buf, &start_time);
		std::strftime(timedisplay, sizeof(timedisplay), "_%Y_%m_%d__%H_%M_%S.csv", &buf);
		return std::string(timedisplay);
	}

	/*
	 * Returns a datetime stamp WITHOUT the .csv extension, for sharing
	 * between traj/gains/scores files so they can be paired up by name.
	 */
	std::string get_datetime_stem() {
		std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
		std::time_t start_time = std::chrono::system_clock::to_time_t(now);
		char timedisplay[100];
		tm buf;
		errno_t err = localtime_s(&buf, &start_time);
		std::strftime(timedisplay, sizeof(timedisplay), "_%Y_%m_%d__%H_%M_%S", &buf);
		return std::string(timedisplay);
	}


	void save_traj_bundle(int state_dim, int num_timesteps, int num_trajectories,
	                       std::vector<float> x_traj, std::string system_name)
	{
		auto start = std::chrono::system_clock::now();
		auto trajectory_path = create_save_location(system_name);
		auto datetime = get_datetime();
		std::ofstream traj_file(trajectory_path + system_name + datetime);
		traj_file << state_dim << ", " << num_timesteps << ", " << num_trajectories << ", ";
		for (int i = 0; i < state_dim * num_timesteps * num_trajectories; ++i) {
			traj_file << x_traj[i];
			if (i == state_dim * num_timesteps * num_trajectories) {
				traj_file << "\n";
			}
			else {
				traj_file << ", ";
			}
		}
		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Trajectory Export Time: " << elapsed.count() / 1000.0 << " seconds." << std::endl;
	}

	/*
	 * Save trajectory bundle + gains using a CALLER-PROVIDED datetime stem
	 * and trajectory directory. Lets the caller share the same stem with
	 * additional files (e.g., a scores CSV).
	 *
	 * Outputs:
	 *   <trajectory_path><system>_<stem>.csv         (trajectories)
	 *   <trajectory_path><system>_<stem>_gains.csv   (gains)
	 */
	template<class GAIN_T>
	void save_traj_and_gains_with_stem(int state_dim, int num_timesteps, int num_trajectories,
	                                    const std::vector<float>& x_traj,
	                                    const std::vector<GAIN_T>& gains,
	                                    const std::string& system_name,
	                                    const std::string& trajectory_path,
	                                    const std::string& stem)
	{
		auto start = std::chrono::system_clock::now();

		std::ofstream traj_file(trajectory_path + system_name + stem + ".csv");
		traj_file << state_dim << ", " << num_timesteps << ", " << num_trajectories << ", ";
		for (int i = 0; i < state_dim * num_timesteps * num_trajectories; ++i) {
			traj_file << x_traj[i];
			traj_file << ((i == state_dim * num_timesteps * num_trajectories) ? "\n" : ", ");
		}
		traj_file.close();

		std::ofstream gains_file(trajectory_path + system_name + stem + "_gains.csv");
		gains_file << "rollout,"
		           << "kp_z,ki_z,kd_z,"
		           << "kp_x,ki_x,kd_x,"
		           << "kp_y,ki_y,kd_y,"
		           << "kp_phi,ki_phi,kd_phi,"
		           << "kp_theta,ki_theta,kd_theta,"
		           << "kp_psi,ki_psi,kd_psi\n";
		for (int r = 0; r < (int)gains.size(); ++r) {
			gains_file << r;
			for (int p = 0; p < 6; ++p) {
				gains_file << "," << gains[r].kp[p]
				           << "," << gains[r].ki[p]
				           << "," << gains[r].kd[p];
			}
			gains_file << "\n";
		}
		gains_file.close();

		auto end = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Trajectory + Gains Export Time: "
		          << elapsed.count() / 1000.0 << " seconds." << std::endl;
		std::cout << "  Trajectories: " << system_name + stem + ".csv" << std::endl;
		std::cout << "  Gains:        " << system_name + stem + "_gains.csv" << std::endl;
	}

	/*
	 * Convenience overload: auto-generates the stem.
	 */
	template<class GAIN_T>
	void save_traj_and_gains(int state_dim, int num_timesteps, int num_trajectories,
	                          const std::vector<float>& x_traj,
	                          const std::vector<GAIN_T>& gains,
	                          const std::string& system_name)
	{
		auto stem = get_datetime_stem();
		auto trajdir = create_save_location(system_name);
		save_traj_and_gains_with_stem(state_dim, num_timesteps, num_trajectories,
		                              x_traj, gains, system_name, trajdir, stem);
	}

}

#endif
