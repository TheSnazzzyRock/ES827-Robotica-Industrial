## @file Robot.py
#  @brief Data container for the robot's physical model.
#  @author Grupo 1
#  @date June 16, 2025
#  @version 1.0
#
#  This file defines the Robot class, which encapsulates all physical properties
#  of the robot, including Standard Denavit-Hartenberg (DH) parameters, link masses,
#  centers of mass, and inertia tensors for the Fanuc LR Mate 200iB robot.

import numpy as np
from sympy import Matrix, symbols

# --- Symbolic variables for use in Kinematics class ---
q1, q2, q3, q4, q5, q6 = symbols("q1 q2 q3 q4 q5 q6")
a1, a2, a3, a4, a5, a6 = symbols("a1 a2 a3 a4 a5 a6")
d1, d2, d3, d4, d5, d6 = symbols("d1 d2 d3 d4 d5 d6")
alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols("alpha1 alpha2 alpha3 alpha4 alpha5 alpha6")

class Robot:
    """
    Data container class for all robot parameters.
    
    This class holds the kinematic (Standard DH) and dynamic (mass, inertia)
    parameters for the Fanuc LR Mate 200iB robot. It provides these parameters
    to the Kinematics, Dynamics, and Control modules.
    """
    def __init__(self, s_name: str = "Fanuc LR Mate 200iB"):
        """
        Initializes the robot model with all its parameters.
        @param s_name: The name of the robot.
        """
        self.s_name: str = s_name
        self.i_num_joints: int = 6

        # --- Symbolic Definitions for Kinematics Module ---
        self.list_joint_vars: list[symbols] = [q1, q2, q3, q4, q5, q6]
        self.list_dh_params_sym: list[list[symbols]] = [
            [a1, alpha1, d1, q1], [a2, alpha2, d2, q2], [a3, alpha3, d3, q3],
            [a4, alpha4, d4, q4], [a5, alpha5, d5, q5], [a6, alpha6, d6, q6],
        ]

        # --- Numerical Kinematic Parameters (Standard DH) ---
        # Source: Peter Corke's Robotics Toolbox (a verified canonical source)
        self.list_numerical_a: list[float] = [0.0, 0.290, 0.075, 0.0, 0.0, 0.0]
        self.list_numerical_alpha: list[float] = [np.pi/2, 0.0, np.pi/2, -np.pi/2, np.pi/2, 0.0]
        self.list_numerical_d: list[float] = [0.330, 0.0, 0.0, -0.320, 0.0, -0.080]

        # --- Numerical Dynamic Parameters (Extracted from MATLAB project) ---
        self.list_numerical_masses: list[float] = [
            5.7031, 2.7230, 1.3000, 0.5310, 0.3340, 0.1060
        ]
        self.list_numerical_com_vectors: list[np.ndarray] = [
            np.array([-0.0000, -0.0163, 0.0382]), np.array([0.1652, 0.0001, -0.0041]),
            np.array([0.0108, 0.0000, 0.0016]), np.array([0.0000, 0.0001, 0.1147]),
            np.array([-0.0001, -0.0123, -0.0001]), np.array([0.1458, -0.0001, 0.0000]),
        ]
        inertia_vectors = [
            [0.015024, 0.006232, 0.014527, 0.000007, -0.000021, -0.000435],
            [0.002678, 0.043567, 0.044627, -0.000002, 0.003189, 0.000008],
            [0.001155, 0.001242, 0.000407, 0.000000, -0.000003, 0.000000],
            [0.001377, 0.001402, 0.000132, 0.000000, -0.000000, 0.000003],
            [0.000185, 0.000117, 0.000186, 0.000000, 0.000000, -0.000013],
            [0.000063, 0.000201, 0.000204, 0.000000, -0.000000, 0.000000]
        ]
        self.list_numerical_inertia_tensors: list[np.ndarray] = []
        for iv in inertia_vectors:
            Ixx, Iyy, Izz, Ixy, Ixz, Iyz = iv
            self.list_numerical_inertia_tensors.append(np.array([
                [Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]
            ]))

        self.mat_gravity_vector: Matrix = Matrix([0, 0, -9.81])