## @file main.py
#  @brief Main script for simulating the Niryo One robot with a PID controller.
#         Demonstrates kinematic, dynamic, and control functionalities.
#  @author Grupo 1
#  @date June 7, 2025
#
#  This script orchestrates the simulation of the Niryo One robot. It initializes
#  the robot's model, kinematics, dynamics, and control system. It then performs
#  various tests including symbolic/numerical forward kinematics, inverse kinematics,
#  inverse dynamics, and a closed-loop trajectory tracking simulation using a PID controller.

import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix, cos, pi, pretty_print, sin

from Control import Control
from Dynamics import Dynamics
from Kinematics import Kinematics
from Robot import Robot

if __name__ == "__main__":
    # Initialize the robot model
    obj_niryo_one_model: Robot = Robot()

    # Initialize the kinematics calculator
    obj_kinematics_calculator: Kinematics = Kinematics(obj_niryo_one_model)

    # Initialize the dynamics calculator
    obj_dynamics_calculator: Dynamics = Dynamics(
        obj_niryo_one_model, obj_kinematics_calculator
    )

    # Initialize the control system
    obj_control_system: Control = Control(obj_niryo_one_model, obj_dynamics_calculator)
    print(
        "Calculated PID Gains: Kp={0}, Ki={1}, Kd={2}".format(
            obj_control_system.f_kp,
            obj_control_system.f_ki,
            obj_control_system.f_kd,
        )
    )

    # --- Symbolic Forward Kinematics ---
    print("\n--- Symbolic Forward Kinematics ---")
    mat_t0_6_symbolic: Matrix = obj_kinematics_calculator.symbolic_fk_matrix
    pretty_print(mat_t0_6_symbolic)
    print("\n")

    # --- Numerical Forward Kinematics ---
    print("--- Numerical Forward Kinematics ---")
    list_joint_angles_example: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mat_t0_6_numerical: Matrix = obj_kinematics_calculator.calculate_numerical_fk(
        list_joint_angles_example
    )
    pretty_print(mat_t0_6_numerical.evalf(subs={pi: np.pi}))

    # Extract position and orientation of the end-effector
    mat_position: Matrix = mat_t0_6_numerical[:3, 3]
    mat_orientation: Matrix = mat_t0_6_numerical[:3, :3]

    print(
        "End-effector's final position (x, y, z): {0}".format(
            mat_position.evalf(subs={pi: np.pi})
        )
    )
    print("Final orientation matrix:\n")
    pretty_print(mat_orientation.evalf(subs={pi: np.pi}))

    # --- Inverse Kinematics ---
    print("\n--- Inverse Kinematics ---")
    mat_target_pose_example: Matrix = mat_t0_6_numerical.evalf(subs={pi: np.pi})

    # Call the inverse kinematics function
    list_found_angles_ik: list[float] = (
        obj_kinematics_calculator.calculate_inverse_kinematics(mat_target_pose_example)
    )
    print("Found Angles (IK Scipy): {0}".format(list_found_angles_ik))

    # --- Dynamics ---
    print("\n--- Dynamics ---")
    list_q_test_dyn: list[float] = [0.0, np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0]
    list_q_dot_test_dyn: list[float] = [0.0, 0.1, 0.1, 0.0, 0.0, 0.0]
    list_q_ddot_test_dyn: list[float] = [0.0, 0.01, 0.01, 0.0, 0.0, 0.0]
    list_external_force_at_tool: list[float] = [0.0, 0.0, -10.0, 0.0, 0.0, 0.0]

    list_required_torques_id: list[float] = (
        obj_dynamics_calculator.calculate_inverse_dynamics_numerical(
            list_q_test_dyn,
            list_q_dot_test_dyn,
            list_q_ddot_test_dyn,
            list_external_force_at_tool,
        )
    )
    print(
        "Required Joint Torques (Inverse Dynamics) (Nm): {0}".format(
            list_required_torques_id
        )
    )

    # --- Forward Dynamics and M, C, G Testing ---
    print("\n--- Forward Dynamics and M, C, G Testing ---")
    mat_m_matrix, f_c_vector, f_g_vector = (
        obj_dynamics_calculator.get_mass_matrix_and_coriolis_gravity(
            list_q_test_dyn, list_q_dot_test_dyn
        )
    )
    print("Mass Matrix (M):\n{0}".format(mat_m_matrix))
    print("Coriolis/Centrifugal Terms Vector (C):\n{0}".format(f_c_vector))
    print("Gravity Terms Vector (G):\n{0}".format(f_g_vector))

    list_calculated_q_ddot_fd: list[float] = (
        obj_dynamics_calculator.forward_dynamics_numerical(
            list_q_test_dyn,
            list_q_dot_test_dyn,
            list_required_torques_id,
            list_external_force_at_tool=list_external_force_at_tool,
        )
    )
    print("Calculated q_ddot (Forward Dynamics): {0}".format(list_calculated_q_ddot_fd))
    print("Original q_ddot_test_dyn: {0}".format(list_q_ddot_test_dyn))

    # --- Trajectory Control Simulation (Closed-Loop PID) ---
    print("\n--- Trajectory Control Simulation (PID) ---")

    # Simulation parameters
    f_total_sim_time: float = 5.0
    f_dt: float = obj_control_system.f_dt
    i_num_simulation_steps: int = int(f_total_sim_time / f_dt)

    # Define desired trajectory for all joints
    f_q_initial: np.ndarray = np.array([0.0] * obj_niryo_one_model.i_num_joints)
    # Move Joint 2 from 0 to pi/2 radians
    f_q_target: np.ndarray = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0])

    # Lists to store simulation data for plotting
    list_time_points: list[float] = []
    list_q_desired_history: list[list[float]] = []
    list_q_dot_desired_history: list[list[float]] = []
    list_q_current_history: list[list[float]] = []
    list_q_dot_current_history: list[list[float]] = []
    list_tau_command_history: list[list[float]] = []
    list_error_history: list[list[float]] = []

    # Current robot state starting at home position
    f_q_current_sim: np.ndarray = f_q_initial.copy()
    f_q_dot_current_sim: np.ndarray = np.array([0.0] * obj_niryo_one_model.i_num_joints)

    # Add initial state to history lists before the simulation loop
    list_time_points.append(0.0)
    list_q_desired_history.append(f_q_initial.tolist())
    list_q_dot_desired_history.append(
        np.array([0.0] * obj_niryo_one_model.i_num_joints).tolist()
    )
    list_q_current_history.append(f_q_current_sim.tolist())
    list_q_dot_current_history.append(f_q_dot_current_sim.tolist())
    list_error_history.append(
        (f_q_initial - f_q_current_sim).tolist()
    )  # Initial error (0.0)

    print(
        "Simulating for {0} seconds with dt = {1} s...".format(f_total_sim_time, f_dt)
    )

    # Main simulation loop
    for i_step in range(1, i_num_simulation_steps + 1):
        f_current_time: float = i_step * f_dt
        list_time_points.append(f_current_time)

        # Define desired position and velocity at current time
        # Linear trajectory from 0 to 5 seconds, then maintain target position
        if f_current_time < 5.0:
            f_q_des_step: np.ndarray = f_q_initial + (f_q_target - f_q_initial) * (
                f_current_time / 5.0
            )
            f_q_dot_des_step: np.ndarray = (f_q_target - f_q_initial) / 2.0
        else:
            f_q_des_step = f_q_target
            f_q_dot_des_step = np.array([0.0] * obj_niryo_one_model.i_num_joints)

        list_q_desired_history.append(f_q_des_step.tolist())
        list_q_dot_desired_history.append(f_q_dot_des_step.tolist())

        # Calculate command torques using the PID controller
        list_command_torques: list[float] = obj_control_system.calculate_pid_torques(
            f_q_des_step.tolist(),  # Desired position
            f_q_dot_des_step.tolist(),  # Desired velocity
            f_q_current_sim.tolist(),  # Current position
            f_q_dot_current_sim.tolist(),  # Current velocity
        )
        list_tau_command_history.append(list_command_torques)

        # Calculate acceleration using the robot's forward dynamics
        list_q_ddot_sim: list[float] = (
            obj_dynamics_calculator.forward_dynamics_numerical(
                f_q_current_sim.tolist(),
                f_q_dot_current_sim.tolist(),
                list_command_torques,
                list_external_force_at_tool=None,
            )
        )

        # 4. Integrate to get new velocities and positions
        # q_dot(t+dt) = q_dot(t) + q_ddot(t) * dt
        f_q_dot_current_sim += np.array(list_q_ddot_sim) * f_dt
        # q(t+dt) = q(t) + q_dot(t+dt) * dt
        f_q_current_sim += f_q_dot_current_sim * f_dt

        # Store current state for plotting
        list_q_current_history.append(f_q_current_sim.tolist())
        list_q_dot_current_history.append(f_q_dot_current_sim.tolist())
        list_error_history.append((f_q_des_step - f_q_current_sim).tolist())

    print("Simulation completed. Verifying data before plotting...")

    # --- Convert history lists to NumPy arrays before any .shape access ---
    mat_q_desired_history: np.ndarray = np.array(list_q_desired_history)
    mat_q_current_history: np.ndarray = np.array(list_q_current_history)
    mat_q_dot_desired_history: np.ndarray = np.array(list_q_dot_desired_history)
    mat_q_dot_current_history: np.ndarray = np.array(list_q_dot_current_history)
    mat_tau_command_history: np.ndarray = np.array(list_tau_command_history)
    mat_error_history: np.ndarray = np.array(list_error_history)

    # --- Plotting Results ---
    plt.figure(figsize=(15, 12))

    # Plot Joint Positions
    plt.subplot(4, 1, 1)
    for i in range(obj_niryo_one_model.i_num_joints):
        plt.plot(
            list_time_points,
            mat_q_desired_history[:, i],
            linestyle="--",
            label="Joint {0} Desired".format(i + 1),
        )
        plt.plot(
            list_time_points,
            mat_q_current_history[:, i],
            label="Joint {0} Actual".format(i + 1),
        )
    plt.title("Desired vs. Actual Joint Positions")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend(ncol=2)

    # Plot Joint Velocities
    plt.subplot(4, 1, 2)
    for i in range(obj_niryo_one_model.i_num_joints):
        plt.plot(
            list_time_points,
            mat_q_dot_desired_history[:, i],
            linestyle="--",
            label="Joint {0} Desired".format(i + 1),
        )
        plt.plot(
            list_time_points,
            mat_q_dot_current_history[:, i],
            label="Joint {0} Actual".format(i + 1),
        )
    plt.title("Desired vs. Actual Joint Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.grid(True)
    plt.legend(ncol=2)

    # Plot Command Torques
    plt.subplot(4, 1, 3)
    for i in range(obj_niryo_one_model.i_num_joints):
        plt.plot(
            list_time_points[1:],
            mat_tau_command_history[:, i],
            label="Joint {0} Torque".format(i + 1),
        )
    plt.title("Joint Command Torques")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.grid(True)
    plt.legend(ncol=2)

    # Plot Position Error
    plt.subplot(4, 1, 4)
    for i in range(obj_niryo_one_model.i_num_joints):
        plt.plot(
            list_time_points,
            mat_error_history[:, i],
            label="Joint {0} Error".format(i + 1),
        )
    plt.title("Joint Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.grid(True)
    plt.legend(ncol=2)

    plt.tight_layout()
    plt.show()
