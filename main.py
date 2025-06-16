## @file main.py
#  @brief Main script to run the full robot simulation in an object-oriented structure.
#  @author Grupo 1
#  @date June 16, 2025
#  @version 1.0
#
#  This script orchestrates the entire robot simulation. It initializes all necessary
#  modules (Robot, Kinematics, Dynamics, Control), runs a closed-loop trajectory
#  tracking simulation, and plots the results for analysis. It also prints key
#  matrices and vectors for the final report.

import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix, pi, pretty_print
from Control import Control
from Dynamics import Dynamics
from Kinematics import Kinematics
from Robot import Robot

if __name__ == "__main__":
    # --- Initialization ---
    obj_robot_model = Robot()
    obj_kinematics_calculator = Kinematics(obj_robot_model)
    obj_dynamics_calculator = Dynamics(obj_robot_model, obj_kinematics_calculator)
    obj_control_system = Control(obj_robot_model, obj_dynamics_calculator)
    
    # --- Print Initial Information for Report ---
    print("\n--- SYSTEM AND CONTROL INFORMATION ---")
    print(f"Calculated Kp gains: {np.round(obj_control_system.f_kp, 2)}")
    print(f"Calculated Kd gains: {np.round(obj_control_system.f_kd, 2)}")
    print(f"Calculated Ki gains: {np.round(obj_control_system.f_ki, 2)}")

    # --- Simulation Setup ---
    waypoints = np.array([
        [1.1912, -0.0579, -0.3095, -1.4718, 1.2037, 1.3008],
        [1.1912,  0.2385, -0.7305, -1.2531, 1.3594, 0.5678],
        [1.9680,  1.1486,  1.3177,  1.6260, 1.1733, 1.4001],
        [1.9706,  1.4851,  1.3177,  1.4863, 1.1789, 1.7642]
    ])
    
    time_per_segment = 2.0
    dt = obj_control_system.f_dt
    total_sim_time = (len(waypoints) - 1) * time_per_segment
    num_steps = int(total_sim_time / dt)

    # --- History lists for plotting ---
    time_hist = np.zeros(num_steps + 1)
    q_des_hist, q_hist = np.zeros((num_steps + 1, 6)), np.zeros((num_steps + 1, 6))
    qd_des_hist, qd_hist = np.zeros((num_steps + 1, 6)), np.zeros((num_steps + 1, 6))
    tau_hist = np.zeros((num_steps, 6))

    # --- Initial Conditions ---
    q_current = waypoints[0].copy()
    qd_current = np.zeros(6)
    q_hist[0, :], q_des_hist[0, :] = q_current, q_current
    
    print(f"\n--- STARTING SIMULATION ---")
    print(f"Simulating for {total_sim_time:.1f} seconds (dt = {dt}s)...")

    # --- Main Simulation Loop ---
    for i in range(num_steps):
        current_time = i * dt
        time_hist[i+1] = current_time + dt
        
        segment_idx = min(int(current_time / time_per_segment), len(waypoints) - 2)
        time_in_segment = current_time - segment_idx * time_per_segment
        q_start, q_end = waypoints[segment_idx], waypoints[segment_idx + 1]
        
        tf = time_per_segment; t = time_in_segment
        q_des = q_start + (q_end - q_start) * (10*(t/tf)**3 - 15*(t/tf)**4 + 6*(t/tf)**5)
        qd_des = (q_end - q_start) * (30*(t**2)/(tf**3) - 60*(t**3)/(tf**4) + 30*(t**4)/(tf**5))
        
        q_des_hist[i+1, :], qd_des_hist[i+1, :] = q_des, qd_des
        
        # Controller calculates torque
        tau_cmd = obj_control_system.calculate_pid_torques(q_des, qd_des, q_current, qd_current)
        tau_hist[i, :] = tau_cmd
        
        # Forward Dynamics calculates acceleration
        qdd_current = obj_dynamics_calculator.forward_dynamics_numerical(q_current, qd_current, tau_cmd)
        
        # Integrator updates state
        qd_current += np.array(qdd_current) * dt
        q_current += qd_current * dt
        
        q_hist[i+1, :], qd_hist[i+1, :] = q_current, qd_current
        
    print("Simulation completed successfully!")

    print("\n\n--- DATA FOR THE REPORT ---")
    
    # Analysis point 
    report_step = int(num_steps / 2)
    q_report = q_hist[report_step]
    qd_report = qd_hist[report_step]
    tau_report = tau_hist[report_step -1]
    
    print(f"\nAnalysis at time t = {total_sim_time/2:.2f}s")
    print(f"Configuration q: {np.round(q_report, 4)}")
    print(f"Velocity q_dot: {np.round(qd_report, 4)}")
    
    # M, C, G matrices at this instant
    M, C, G = obj_dynamics_calculator.get_mass_matrix_and_coriolis_gravity(q_report.tolist(), qd_report.tolist())
    np.set_printoptions(precision=4, suppress=True)
    print("\nMass Matrix M(q):")
    print(M)
    print("\nCoriolis C(q, q_dot):")
    print(C)
    print("\nGravity Vector G(q):")
    print(G)
    print("\nCommand Torque calculated at this instant:")
    print(np.round(tau_report,4))
    
    # Kinematics and Jacobian at this instant
    T_report = obj_kinematics_calculator.calculate_numerical_fk(q_report.tolist())
    J_report = obj_kinematics_calculator.calculate_numerical_jacobian(q_report.tolist())
    print("\nHomogeneous Transformation Matrix T(q) at this instant:")
    pretty_print(T_report.evalf(subs={pi: np.pi}, n=4))
    print("\nJacobian Matrix J(q) at this instant:")
    print(J_report)
    print("\n" + "="*50 + "\n")



    # plot
    print("Generating graphs...")
    fig, axs = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.suptitle("Fanuc LR Mate 200iB Simulation", fontsize=16)

    axs[0].plot(time_hist, q_hist, lw=2)
    axs[0].plot(time_hist, q_des_hist, '--', lw=1.5)
    axs[0].set_title("Joint Positions")
    axs[0].set_ylabel("Angle (rad)")
    axs[0].grid(True)
    axs[0].legend([f'J{i+1} Real' for i in range(6)] + [f'J{i} Desired' for i in range(1,7)], ncol=4, fontsize='small')

    error = q_des_hist - q_hist
    axs[1].plot(time_hist, error)
    axs[1].set_title("Position Tracking Error")
    axs[1].set_ylabel("Error (rad)")
    axs[1].grid(True)
    axs[1].legend([f'Error J{i+1}' for i in range(6)], ncol=3)

    axs[2].plot(time_hist[:-1], tau_hist)
    axs[2].set_title("Command Torques")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Torque (Nm)")
    axs[2].grid(True)
    axs[2].legend([f'J{i+1} Torque' for i in range(6)], ncol=3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()