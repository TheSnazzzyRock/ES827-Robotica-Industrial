## @file Robot.py
#  @brief Implements the Robot class to store kinematic and dynamic parameters of a manipulator.
#         Specifically configured for the Niryo One robot.
#  @author Grupo 1
#  @date June 7, 2025
#
#  This file defines the Robot class, which encapsulates all physical and dynamic properties
#  of the robot, including Denavit-Hartenberg (DH) parameters, link masses,
#  centers of mass, inertia tensors, and motor parameters.
#  It serves as a data container for the robot's physical model.

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
    rc1x,
    rc1y,
    rc1z,
    rc2x,
    rc2y,
    rc2z,
    rc3x,
    rc3y,
    rc3z,
    rc4x,
    rc4y,
    rc4z,
    rc5x,
    rc5y,
    rc5z,
    rc6x,
    rc6y,
    rc6z,
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
    This class is specifically configured for the Niryo One robot.
    """

    def __init__(self, s_name: str = "Niryo One"):
        """
        Initializes the robot model with its kinematic and dynamic parameters.

        @param s_name: The name of the robot.
        """
        self.s_name: str = s_name
        self.i_num_joints: int = 6

        # --- Kinematic Parameters (DH) ---
        # List of symbolic DH parameters for each joint
        self.list_dh_params_sym: list[list[symbols]] = [
            [a1, alpha1, d1, q1],
            [a2, alpha2, d2, q2],
            [a3, alpha3, d3, q3],
            [a4, alpha4, d4, q4],
            [a5, alpha5, d5, q5],
            [a6, alpha6, d6, q6],
        ]
        # List of joint variable symbols
        self.list_joint_vars: list[symbols] = [q1, q2, q3, q4, q5, q6]

        # Numerical DH parameters
        self.list_numerical_a: list[float] = [
            0,
            640,
            0,
            0,
            0,
            0,
        ]
        self.list_numerical_alpha: list[float] = [
            np.pi / 2,
            0,
            np.pi / 2,
            -np.pi / 2,
            np.pi / 2,
            0,
        ]
        self.list_numerical_d: list[float] = [
            0,
            0,
            0,
            496,
            0,
            75,
        ]

        # --- Dynamic Parameters of the Links ---
        # Link masses (kg)
        self.list_masses_sym: list[symbols] = [m1, m2, m3, m4, m5, m6]

        # Numerical masses
        # SUBSTITUIR PELOS VALORES REAIS!!!!!!!
        self.list_numerical_masses: list[float] = [
            0.5,
            0.3,
            0.2,
            0.1,
            0.05,
            0.02,
        ]

        # Center of mass vectors for each link (x, y, z)
        self.list_com_vectors_sym: list[Matrix] = [
            Matrix([rc1x, rc1y, rc1z]),
            Matrix([rc2x, rc2y, rc2z]),
            Matrix([rc3x, rc3y, rc3z]),
            Matrix([rc4x, rc4y, rc4z]),
            Matrix([rc5x, rc5y, rc5z]),
            Matrix([rc6x, rc6y, rc6z]),
        ]
        # Numerical COM vectors
        # SUBSTITUIR PELOS VALORES REAIS!!!!!!!
        self.list_numerical_com_vectors: list[np.ndarray] = [
            np.array([0.0, 0.0, 0.02]),
            np.array([0.1, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.01]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        ]

        # Inertia Tensors for each link
        self.list_inertia_tensors_sym: list[Matrix] = [
            Matrix([[I1xx, I1xy, I1xz], [I1xy, I1yy, I1yz], [I1xz, I1yz, I1zz]]),
            Matrix([[I2xx, I2xy, I2xz], [I2xy, I2yy, I2yz], [I2xz, I2yz, I2zz]]),
            Matrix([[I3xx, I3xy, I3xz], [I3xy, I3yy, I3yz], [I3xz, I3yz, I3zz]]),
            Matrix([[I4xx, I4xy, I4xz], [I4xy, I4yy, I4yz], [I4xz, I4yz, I4zz]]),
            Matrix([[I5xx, I5xy, I5xz], [I5xy, I5yy, I5yz], [I5xz, I5yz, I5zz]]),
            Matrix([[I6xx, I6xy, I6xz], [I6xy, I6yy, I6yz], [I6xz, I6yz, I6zz]]),
        ]
        # Numerical Inertia Tensors
        # SUBSTITUIR PELOS VALORES REAIS!!!!!!!
        self.list_numerical_inertia_tensors: list[np.ndarray] = [
            np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]),
            np.array([[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0.0001]]),
            np.array([[0.005, 0, 0], [0, 0.005, 0], [0, 0, 0.0001]]),
            np.array([[0.0005, 0, 0], [0, 0.0005, 0], [0, 0, 0.00001]]),
            np.array([[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.00001]]),
            np.array([[0.00005, 0, 0], [0, 0.00005, 0], [0, 0, 0.000001]]),
        ]

        # Gravity vector in the base frame
        self.mat_gravity_vector: Matrix = Matrix([0, 0, -9.81])

        # --- Actuator Parameters (DC Motors for SISO model) ---
        # SUBSTITUIR PELOS VALORES REAIS!!!!!!!
        self.f_motor_kt: float = 0.1
        self.f_motor_kb: float = 0.1
        self.f_motor_r: float = 2.0
        self.f_motor_bm_internal: float = 0.01
        self.f_gear_ratio: float = 100.0
