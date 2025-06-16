## @file Robot.py
#  @brief Implements the Robot class to store kinematic and dynamic parameters.
#         Specifically configured for the Fanuc LR Mate 200iB robot.
#  @author Grupo 1
#  @date June 16, 2025
#
#  This file defines the Robot class, which encapsulates all physical and dynamic properties
#  of the robot, including Denavit-Hartenberg (DH) parameters, link masses,
#  centers of mass, inertia tensors, and motor parameters, extracted from a
#  MATLAB Robotics System Toolbox model.

import numpy as np
from sympy import Matrix, cos, pi, sin, symbols


# Symbolic joint variables (rotation angles)
q1, q2, q3, q4, q5, q6 = symbols("q1 q2 q3 q4 q5 q6")

# Symbolic Denavit-Hartenberg (DH) parameters for each link
a1, a2, a3, a4, a5, a6 = symbols("a1 a2 a3 a4 a5 a6")
d1, d2, d3, d4, d5, d6 = symbols("d1 d2 d3 d4 d5 d6")
alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols(
    "alpha1 alpha2 alpha3 alpha4 alpha5 alpha6"
)

# Symbolic dynamic parameters: mass and inertia components for each link
m1, m2, m3, m4, m5, m6 = symbols("m1 m2 m3 m4 m5 m6")
# Components of the center of mass vector (x, y, z) for each link
(
    rc1x, rc1y, rc1z, rc2x, rc2y, rc2z, rc3x, rc3y, rc3z,
    rc4x, rc4y, rc4z, rc5x, rc5y, rc5z, rc6x, rc6y, rc6z,
) = symbols(
    "rc1x rc1y rc1z rc2x rc2y rc2z rc3x rc3y rc3z rc4x rc4y rc4z rc5x rc5y rc5z rc6x rc6y rc6z"
)
# Components of the inertia matrix for each link
I1xx, I1yy, I1zz, I1xy, I1xz, I1yz = symbols("I1xx I1yy I1zz I1xy I1xz I1yz")
I2xx, I2yy, I2zz, I2xy, I2xz, I2yz = symbols("I2xx I2yy I2zz I2xy I2xz I2yz")
I3xx, I3yy, I3zz, I3xy, I3xz, I3yz = symbols("I3xx I3yy I3zz I3xy I3xz I3yz")
I4xx, I4yy, I4zz, I4xy, I4xz, I4yz = symbols("I4xx I4yy I4zz I4xy I4xz I4yz")
I5xx, I5yy, I5zz, I5xy, I5xz, I5yz = symbols("I5xx I5yy I5zz I5xy I5xz I5yz")
I6xx, I6yy, I6zz, I6xy, I6xz, I6yz = symbols("I6xx I6yy I6zz I6xy I6xz I6yz")


class Robot:
    """
    Represents a robotic manipulator, storing its physical and dynamic parameters.
    This class is specifically configured for the Fanuc LR Mate 200iB robot.
    """

    def __init__(self, s_name: str = "Fanuc LR Mate 200iB"):
        """
        Initializes the robot model with its kinematic and dynamic parameters.

        @param s_name: The name of the robot.
        """
        self.s_name: str = s_name
        self.i_num_joints: int = 6

        # --- Kinematic Parameters (DH) ---
        # List of symbolic DH parameters for each joint
        self.list_dh_params_sym: list[list[symbols]] = [
            [a1, alpha1, d1, q1], [a2, alpha2, d2, q2], [a3, alpha3, d3, q3],
            [a4, alpha4, d4, q4], [a5, alpha5, d5, q5], [a6, alpha6, d6, q6],
        ]
        # List of joint variable symbols
        self.list_joint_vars: list[symbols] = [q1, q2, q3, q4, q5, q6]

        # Numerical DH parameters for Fanuc LR Mate 200iB (MathWorks Standard Convention)
        self.list_numerical_a: list[float] = [0.0, 0.350, 0.150, 0.0, 0.0, 0.0]
        self.list_numerical_alpha: list[float] = [-np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0]
        self.list_numerical_d: list[float] = [0.350, 0.0, 0.0, 0.250, 0.075, 0.370]

        # --- Dynamic Parameters of the Links (Extracted from tamanhos.m) ---
        # Link masses (kg) for Links 1-6
        # Corresponds to bodies {link_1} through {link_6} from MATLAB model
        self.list_numerical_masses: list[float] = [
            5.7031, 2.7230, 1.3000, 0.5310, 0.3340, 0.1060
        ]

        # Center of mass vectors for each link [x, y, z] in meters, relative to the link's frame
        self.list_numerical_com_vectors: list[np.ndarray] = [
            np.array([-0.0000, -0.0163, 0.0382]),     # Link 1
            np.array([0.1652, 0.0001, -0.0041]),      # Link 2
            np.array([0.0108, 0.0000, 0.0016]),       # Link 3
            np.array([0.0000, 0.0001, 0.1147]),       # Link 4
            np.array([-0.0001, -0.0123, -0.0001]),    # Link 5
            np.array([0.1458, -0.0001, 0.0000]),      # Link 6
        ]

        # Inertia Tensors for each link, converted to 3x3 matrices
        # Data from MATLAB is a vector [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        inertia_vectors = [
            [0.015024, 0.006232, 0.014527, 0.000007, -0.000021, -0.000435], # Link 1
            [0.002678, 0.043567, 0.044627, -0.000002, 0.003189, 0.000008], # Link 2
            [0.001155, 0.001242, 0.000407, 0.000000, -0.000003, 0.000000], # Link 3
            [0.001377, 0.001402, 0.000132, 0.000000, -0.000000, 0.000003], # Link 4
            [0.000185, 0.000117, 0.000186, 0.000000, 0.000000, -0.000013], # Link 5
            [0.000063, 0.000201, 0.000204, 0.000000, -0.000000, 0.000000]  # Link 6
        ]
        self.list_numerical_inertia_tensors: list[np.ndarray] = []
        for iv in inertia_vectors:
            Ixx, Iyy, Izz, Ixy, Ixz, Iyz = iv
            inertia_matrix = np.array([
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz]
            ])
            self.list_numerical_inertia_tensors.append(inertia_matrix)

        # Gravity vector in the base frame
        self.mat_gravity_vector: Matrix = Matrix([0, 0, -9.81])

        # --- Actuator Parameters (DC Motors for SISO model) ---
        # ATENÇÃO: Estes valores ainda são de exemplo e podem precisar de ajuste
        # para o Fanuc, se houver dados disponíveis nos manuais.
        self.f_motor_kt: float = 0.1
        self.f_motor_kb: float = 0.1
        self.f_motor_r: float = 2.0
        self.f_motor_bm_internal: float = 0.01
        self.f_gear_ratio: float = 100.0