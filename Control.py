## @file Control.py
#  @brief Implements a PID controller with automatic gain tuning.
#  @author Grupo 1
#  @date June 16, 2025
#  @version 1.0
#
#  This file defines the Control class, which calculates the command torques
#  for each robot joint based on a desired trajectory and the robot's current state.
#  It uses a PID control strategy with gains tuned automatically based on the
#  robot's dynamic model.

import numpy as np
from Dynamics import Dynamics
from Robot import Robot

class Control:
    """
    Implements a PID controller for each joint of the robot.
    Gains are tuned automatically based on a desired performance and the
    robot's dynamic model.
    """
    def __init__(self, obj_robot_model: Robot, obj_dynamics_calculator: Dynamics):
        """
        Initializes the control system and tunes the PID gains.
        @param obj_robot_model: The robot's data model.
        @param obj_dynamics_calculator: The robot's dynamics calculator.
        """
        self.robot = obj_robot_model
        self.dynamics = obj_dynamics_calculator
        self.i_num_joints = obj_robot_model.i_num_joints
        self.f_dt: float = 0.001

        # --- Automatic PID Gain Tuning ---
        DESIRED_OMEGA_N = np.array([8.0, 8.0, 8.0, 12.0, 12.0, 12.0])
        DESIRED_XI = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.f_kp = np.zeros(self.i_num_joints)
        self.f_kd = np.zeros(self.i_num_joints)
        self.f_ki = np.zeros(self.i_num_joints)
        
        initial_q = [1.1912, -0.0579, -0.3095, -1.4718, 1.2037, 1.3008]
        mat_m, _, _ = self.dynamics.get_mass_matrix_and_coriolis_gravity(initial_q, [0.0] * self.i_num_joints)

        for i in range(self.i_num_joints):
            f_j_effective = mat_m[i, i]
            if f_j_effective < 1e-6: f_j_effective = 1e-6

            self.f_kp[i] = (DESIRED_OMEGA_N[i]**2) * f_j_effective
            self.f_kd[i] = (2 * DESIRED_XI[i] * DESIRED_OMEGA_N[i]) * f_j_effective
            self.f_ki[i] = self.f_kp[i] * 0.1

        self.f_integral_error = np.zeros(self.i_num_joints)

    def calculate_pid_torques(
        self, q_des: list, qd_des: list, q_curr: list, qd_curr: list
    ) -> list[float]:
        """
        Calculates the command torques for each joint using the PID law.
        @param q_des: Desired joint positions.
        @param qd_des: Desired joint velocities.
        @param q_curr: Current joint positions.
        @param qd_curr: Current joint velocities.
        @return: A list of command torques.
        """
        error_pos = np.array(q_des) - np.array(q_curr)
        error_vel = np.array(qd_des) - np.array(q_curr)
        self.f_integral_error += error_pos * self.f_dt
        
        torques = self.f_kp * error_pos + self.f_kd * error_vel + self.f_ki * self.f_integral_error
        
        max_torque = np.array([150.0] * self.i_num_joints)
        return np.clip(torques, -max_torque, max_torque).tolist()