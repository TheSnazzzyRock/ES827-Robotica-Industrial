## @file Control.py
#  @brief Implements the Control class for robot joint control.
#         Provides a PID (Proportional-Integral-Derivative) controller for each joint.
#  @author Grupo 1
#  @date June 7, 2025
#
#  This file defines the Control class, which calculates the command torques
#  for each robot joint based on a desired trajectory and the robot's current state.
#  It uses a PID control strategy with gains tuned based on a simplified SISO model.

import numpy as np

from Dynamics import Dynamics
from Robot import Robot


class Control:
    """
    Class for implementing robot manipulator controllers.
    Provides a per-joint PID (Proportional-Integral-Derivative) controller.
    """

    def __init__(self, robot_model: Robot, dynamics_calculator: Dynamics = None):
        """
        Initializes the control system with the robot model and, optionally,
        a dynamics calculator for model-based gain tuning.

        @param robot_model: An instance of the Robot class.
        @param dynamics_calculator: An optional instance of the Dynamics class.
                                   Used to infer dynamic parameters for PID gain tuning.
        """
        self.robot = robot_model
        self.dynamics = dynamics_calculator
        self.i_num_joints: int = robot_model.i_num_joints

        # Robot's motor and transmission parameters
        # Assumed to be scalar if all motors are identical.
        f_kt: float = self.robot.f_motor_kt
        f_kb: float = self.robot.f_motor_kb
        f_r_motor: float = self.robot.f_motor_r
        f_bm_internal: float = self.robot.f_motor_bm_internal

        # Desired performance parameters for each joint for PID tuning
        # Natural frequency (f_omega_n, rad/s): Determines response speed.
        # Damping ratio (f_xi): Controls response damping (1.0 for critically damped, no overshoot).
        self.f_desired_omega_n: np.ndarray = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.f_desired_xi: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Calculated PID gains (initialized with zeros)
        self.f_kp: np.ndarray = np.zeros(self.i_num_joints)
        self.f_kd: np.ndarray = np.zeros(self.i_num_joints)
        self.f_ki: np.ndarray = np.zeros(self.i_num_joints)

        # Calculate PID gains for each joint using the simplified SISO model
        for i in range(self.i_num_joints):
            # Get the effective joint inertia (f_j_effective).
            # This is an approximation of 'J' in the SISO model control slides from ES827
            # using the diagonal element of the mass matrix M(q) at the zero pose.
            if self.dynamics is not None:
                list_q_zero = [0.0] * self.i_num_joints
                list_q_dot_zero = [0.0] * self.i_num_joints
                mat_m_at_zero, _, _ = (
                    self.dynamics.get_mass_matrix_and_coriolis_gravity(
                        list_q_zero, list_q_dot_zero
                    )
                )
                f_j_effective: float = mat_m_at_zero[i, i]
            else:
                print(
                    f"Warning: Dynamics calculator not provided or unavailable for joint {i+1}. Using default J_effective (0.1)."
                )
                f_j_effective = 0.1

            # Effective joint damping (f_b_effective) in the SISO model.
            # B = B_m_internal + K_b * K_t / R
            f_b_effective: float = f_bm_internal + (f_kb * f_kt / f_r_motor)

            f_omega_n_i: float = self.f_desired_omega_n[i]
            f_xi_i: float = self.f_desired_xi[i]

            # Calculate Kp: Kp = omega_n^2 * J
            self.f_kp[i] = (f_omega_n_i**2) * f_j_effective

            # Calculate Kd: Kd = 2 * xi * omega_n * J - B
            self.f_kd[i] = (2 * f_xi_i * f_omega_n_i * f_j_effective) - f_b_effective

            # Ensure Kd is not negative
            self.f_kd[i] = max(0.0, self.f_kd[i])

            # Calculate Ki: Ki < (B+Kd)*Kp/J
            self.f_ki[i] = self.f_kp[i] / 10.0

        # State variables for the integrator and derivative terms
        self.f_integral_error: np.ndarray = np.zeros(self.i_num_joints)
        self.f_previous_error: np.ndarray = np.zeros(self.i_num_joints)

        # Time interval for derivative calculation (simulation step time)
        self.f_dt: float = 0.01

    def calculate_pid_torques(
        self,
        list_q_desired: list[float],
        list_q_dot_desired: list[float],
        list_q_current: list[float],
        list_q_dot_current: list[float],
    ) -> list[float]:
        """
        Calculates the command torques for each joint using independent PID controllers.

        @param list_q_desired: Desired joint positions in radians.
        @param list_q_dot_desired: Desired joint velocities in rad/s.
        @param list_q_current: Current joint positions in radians (measured).
        @param list_q_dot_current: Current joint velocities in rad/s (measured).
        @return list[float]: The command torques (Nm) to be applied to each joint.
        """
        f_q_desired_np: np.ndarray = np.array(list_q_desired)
        f_q_dot_desired_np: np.ndarray = np.array(list_q_dot_desired)
        f_q_current_np: np.ndarray = np.array(list_q_current)
        f_q_dot_current_np: np.ndarray = np.array(list_q_dot_current)

        # Position error: e(t) = q_desired - q_current
        f_error_pos: np.ndarray = f_q_desired_np - f_q_current_np

        # Velocity error: e_dot(t) = q_dot_desired - q_dot_current
        # Used for the derivative term of the PID
        f_error_vel: np.ndarray = f_q_dot_desired_np - f_q_dot_current_np

        # Proportional term: Kp * e(t)
        f_p_term: np.ndarray = self.f_kp * f_error_pos

        # Integral term: Ki * integral(e(t) dt)
        self.f_integral_error += f_error_pos * self.f_dt
        f_i_term: np.ndarray = self.f_ki * self.f_integral_error

        # Derivative term: Kd * e_dot(t)
        f_d_term: np.ndarray = self.f_kd * f_error_vel

        # Total command torque for each joint: u(t) = P_term + I_term + D_term
        f_command_torques: np.ndarray = f_p_term + f_i_term + f_d_term

        # TROCAR PELO VALOR REAL!!!!!!!
        f_max_torque_per_joint: float = 50.0
        f_command_torques = np.clip(
            f_command_torques, -f_max_torque_per_joint, f_max_torque_per_joint
        )

        return f_command_torques.tolist()
