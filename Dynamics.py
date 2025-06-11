## @file Dynamics.py
#  @brief Implements the Dynamics class for robot dynamic calculations.
#         Provides inverse and forward dynamics functionalities using the Newton-Euler algorithm.
#  @author Grupo 1
#  @date June 7, 2025
#
#  This file defines the Dynamics class, which calculates the torques required
#  for a desired motion (inverse dynamics) and the accelerations resulting
#  from applied torques (forward dynamics). It relies on the Robot model
#  for physical properties and Kinematics for joint transformations.

import numpy as np
from sympy import Matrix, cos, pi, sin, symbols

from Kinematics import Kinematics
from Robot import Robot


class Dynamics:
    """
    Class for performing robot dynamic calculations, including inverse dynamics
    (torques required for motion) and forward dynamics (accelerations resulting
    from applied torques). It implements the Newton-Euler algorithm.
    """

    def __init__(self, robot_model: Robot, kinematics_calculator: Kinematics):
        """
        Initializes the dynamics calculator with the robot model and kinematics calculator.

        @param robot_model: An instance of the Robot class containing the robot's parameters.
        @param kinematics_calculator: An instance of the Kinematics class for
                                       accessing transformations and Jacobian.
        """
        self.robot = robot_model
        self.kinematics = kinematics_calculator

        # Symbolic joint velocities and accelerations (time-varying variables)
        self.list_q_dot_sym: list[symbols] = [
            symbols(f"q{i+1}_dot") for i in range(self.robot.i_num_joints)
        ]
        self.list_q_ddot_sym: list[symbols] = [
            symbols(f"q{i+1}_ddot") for i in range(self.robot.i_num_joints)
        ]

    def _calculate_dh_matrix(
        self, f_a: symbols, f_alpha: symbols, f_d: symbols, f_theta: symbols
    ) -> Matrix:
        """
        Calculates the homogeneous transformation matrix for a single link
        using Denavit-Hartenberg (DH) parameters. This is a helper method,
        duplicated from Kinematics for self-sufficiency and to avoid circular
        dependencies if Kinematics were to depend on Dynamics.

        @param f_a: The link length.
        @param f_alpha: The link twist (in radians).
        @param f_d: The joint offset.
        @param f_theta: The joint angle (in radians).
        @return sympy.Matrix: A 4x4 homogeneous transformation matrix.
        """
        return Matrix(
            [
                [
                    cos(f_theta),
                    -sin(f_theta) * cos(f_alpha),
                    sin(f_theta) * sin(f_alpha),
                    f_a * cos(f_theta),
                ],
                [
                    sin(f_theta),
                    cos(f_theta) * cos(f_alpha),
                    -cos(f_theta) * sin(f_alpha),  # <-- Linha corrigida aqui
                    f_a * sin(f_theta),
                ],
                [0, sin(f_alpha), cos(f_alpha), f_d],
                [0, 0, 0, 1],
            ]
        )

    def calculate_inverse_dynamics_numerical(
        self,
        list_joint_angles: list[float],  # q (joint positions)
        list_joint_velocities: list[float],  # q_dot (joint velocities)
        list_joint_accelerations: list[float],  # q_ddot (joint accelerations)
        list_external_force_at_tool: list[
            float
        ] = None,  # External force/torque at end-effector
    ) -> list[float]:
        """
        Calculates the required joint torques using the Inverse Newton-Euler algorithm.
        Assumes gravity acts in the -Z direction in the robot's base frame.

        @param list_joint_angles: A list of 6 joint angles in radians.
        @param list_joint_velocities: A list of 6 joint angular velocities in rad/s.
        @param list_joint_accelerations: A list of 6 joint angular accelerations in rad/s^2.
        @param list_external_force_at_tool: A list of 6 elements [fx, fy, fz, tx, ty, tz]
                                             representing external force and torque applied
                                             at the end-effector, in the end-effector's frame.
                                             If None, assumes zero external force/torque.
        @return list[float]: The torques [Nm] required at each joint to produce the desired motion.
        """

        f_q: np.ndarray = np.array(list_joint_angles, dtype=np.float64)
        f_q_dot: np.ndarray = np.array(list_joint_velocities, dtype=np.float64)
        f_q_ddot: np.ndarray = np.array(list_joint_accelerations, dtype=np.float64)

        f_ext_tool_np: np.ndarray = (
            np.zeros(6)
            if list_external_force_at_tool is None
            else np.array(list_external_force_at_tool, dtype=np.float64)
        )

        i_num_joints: int = self.robot.i_num_joints

        # Step 1: Forward Recursion (from base to end-effector)
        # Calculate angular and linear velocities and accelerations for each link.
        list_omega: list[np.ndarray] = [np.zeros(3) for _ in range(i_num_joints + 1)]
        list_omega_dot: list[np.ndarray] = [
            np.zeros(3) for _ in range(i_num_joints + 1)
        ]
        list_v: list[np.ndarray] = [np.zeros(3) for _ in range(i_num_joints + 1)]
        list_v_dot: list[np.ndarray] = [np.zeros(3) for _ in range(i_num_joints + 1)]

        # Base frame acceleration due to apparent gravity
        f_a_gravity: np.ndarray = (
            np.array(self.robot.mat_gravity_vector).astype(np.float64).flatten()
        )
        list_v_dot[0] = -f_a_gravity

        # Lists to store rotation matrices (R_i^(i-1)) and position vectors (p_{i-1}^i) between frames
        list_r_curr_to_prev: list[np.ndarray] = [
            np.eye(3) for _ in range(i_num_joints + 1)
        ]
        list_p_prev_to_curr: list[np.ndarray] = [
            np.zeros(3) for _ in range(i_num_joints + 1)
        ]

        # Iterate through each joint from 1 to N
        for i in range(1, i_num_joints + 1):
            # Get numerical DH parameters for link i-1 (associated with joint i)
            f_a_i = self.robot.list_numerical_a[i - 1]
            f_alpha_i = self.robot.list_numerical_alpha[i - 1]
            f_d_i = self.robot.list_numerical_d[i - 1]
            f_theta_i = f_q[i - 1]

            # Calculate homogeneous transformation matrix T_{i-1}^i
            mat_t_link_sympy: Matrix = self._calculate_dh_matrix(
                f_a_i, f_alpha_i, f_d_i, f_theta_i
            )
            mat_t_link_np: np.ndarray = np.array(mat_t_link_sympy).astype(np.float64)

            mat_r_prev_to_curr: np.ndarray = mat_t_link_np[:3, :3]  # Rotation R_{i-1}^i
            f_p_prev_to_curr: np.ndarray = mat_t_link_np[:3, 3]

            # Store rotation from frame i to i-1 (R_i^(i-1)) and position vector p_{i-1}^i
            list_r_curr_to_prev[i] = mat_r_prev_to_curr.T
            list_p_prev_to_curr[i] = f_p_prev_to_curr

            # Angular velocity: omega_i = R_i^(i-1) * omega_{i-1} + z_0 * q_dot_i
            list_omega[i] = list_r_curr_to_prev[i] @ list_omega[i - 1] + np.array(
                [0, 0, f_q_dot[i - 1]]
            )

            # Angular acceleration: omega_dot_i = R_i^(i-1) * omega_dot_{i-1} + omega_i x (z_0 * q_dot_i) + z_0 * q_ddot_i
            list_omega_dot[i] = (
                list_r_curr_to_prev[i] @ list_omega_dot[i - 1]
                + np.cross(list_omega[i], np.array([0, 0, f_q_dot[i - 1]]))
                + np.array([0, 0, f_q_ddot[i - 1]])
            )

            # Linear acceleration: v_dot_i = R_i^(i-1) * v_dot_{i-1} + omega_dot_i x p_{i-1}^i + omega_i x (omega_i x p_{i-1}^i)
            list_v_dot[i] = (
                list_r_curr_to_prev[i] @ list_v_dot[i - 1]
                + np.cross(list_omega_dot[i], list_p_prev_to_curr[i])
                + np.cross(
                    list_omega[i], np.cross(list_omega[i], list_p_prev_to_curr[i])
                )
            )

        # Step 2: Backward Recursion
        # Calculate forces and torques at the joints.
        list_f: list[np.ndarray] = [np.zeros(3) for _ in range(i_num_joints + 1)]
        list_n: list[np.ndarray] = [np.zeros(3) for _ in range(i_num_joints + 1)]

        # Array to store the calculated joint torques
        f_tau: np.ndarray = np.zeros(i_num_joints)

        # External force and torque applied at the end-effector
        list_f[i_num_joints] = f_ext_tool_np[:3]
        list_n[i_num_joints] = f_ext_tool_np[3:]

        # Propagate backwards
        for i in range(i_num_joints, 0, -1):
            f_m_i = self.robot.list_numerical_masses[i - 1]
            f_rc_i = self.robot.list_numerical_com_vectors[i - 1]
            mat_I_i = self.robot.list_numerical_inertia_tensors[
                i - 1
            ]  # Variável local correta

            # Acceleration of link i's center of mass
            f_a_ci = (
                list_v_dot[i]
                + np.cross(list_omega_dot[i], f_rc_i)
                + np.cross(list_omega[i], np.cross(list_omega[i], f_rc_i))
            )

            # Total force on link i (F_i = m_i * a_ci)
            f_f_i = f_m_i * f_a_ci

            # Total torque on link i (N_i = I_i * omega_dot_i + omega_i x (I_i * omega_i))
            f_n_i = mat_I_i @ list_omega_dot[i] + np.cross(
                list_omega[i], mat_I_i @ list_omega[i]  # CORREÇÃO: Usando mat_I_i
            )

            # Force and torque propagated to the previous joint (i-1)
            # f_{i-1} = R_{i-1}^i * f_i + F_i
            list_f[i - 1] = list_r_curr_to_prev[i] @ list_f[i] + f_f_i

            # n_{i-1} = R_{i-1}^i * n_i + N_i + (p_{i-1}^i + r_{c,i}) x F_i
            list_n[i - 1] = (
                list_r_curr_to_prev[i] @ list_n[i]
                + f_n_i
                + np.cross(list_p_prev_to_curr[i] + f_rc_i, f_f_i)
            )

            # Torque at joint i (tau_i)
            # For revolute joints, this is the Z-component of torque n_{i-1}
            # (as the joint axis is aligned with the Z-axis of frame i-1).
            f_tau[i - 1] = list_n[i - 1][2]  # Z-component of torque in frame i-1

        return f_tau.tolist()

    def get_mass_matrix_and_coriolis_gravity(
        self, list_joint_angles: list[float], list_joint_velocities: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the robot's mass matrix (M), Coriolis/centrifugal terms vector (C),
        and gravity terms vector (G) for a given robot configuration.
        These terms are inferred using multiple calls to the Inverse Newton-Euler algorithm.

        @param list_joint_angles: Current joint positions in radians.
        @param list_joint_velocities: Current joint velocities in rad/s.
        @return tuple[np.ndarray, np.ndarray, np.ndarray]:
            - np.ndarray M: Robot's mass matrix (N_joints x N_joints).
            - np.ndarray C: Coriolis and centrifugal terms vector (N_joints x 1).
            - np.ndarray G: Gravity terms vector (N_joints x 1).
        """
        i_num_joints: int = self.robot.i_num_joints

        # Calculate G(q):
        # G is the joint torque required when joint velocities and accelerations are zero.
        list_grav_torques = self.calculate_inverse_dynamics_numerical(
            list_joint_angles,
            [0.0] * i_num_joints,  # q_dot = 0
            [0.0] * i_num_joints,  # q_ddot = 0
            list_external_force_at_tool=np.zeros(6),
        )
        f_g_vec: np.ndarray = np.array(list_grav_torques)

        # 2. Calculate M(q) (mass matrix):
        # The dynamic equation is: M(q) * q_ddot + C(q, q_dot) + G(q) = tau.
        # When q_dot = 0, C = 0. So, M(q) * q_ddot + G(q) = tau.
        # To find column 'j' of M, we set q_ddot to a unit vector in direction 'j'
        # and all other q_dots to zero. Then, M_j_column = tau_j - G.
        mat_m: np.ndarray = np.zeros((i_num_joints, i_num_joints))

        for j in range(i_num_joints):
            f_q_ddot_test = np.zeros(i_num_joints)
            f_q_ddot_test[j] = 1.0

            # Calculate joint torques for this case (q_dot = 0)
            list_tau_col_j = self.calculate_inverse_dynamics_numerical(
                list_joint_angles,
                [0.0] * i_num_joints,
                f_q_ddot_test.tolist(),
                list_external_force_at_tool=np.zeros(6),
            )
            f_tau_col_j_np: np.ndarray = np.array(list_tau_col_j)

            # Column j of the mass matrix is f_tau_col_j_np - G
            mat_m[:, j] = f_tau_col_j_np - f_g_vec

        # 3. Calculate C(q, q_dot) (Coriolis/centrifugal terms):
        # C(q, q_dot) = tau - G (when q_ddot = 0)
        list_coriolis_gravity_torques = self.calculate_inverse_dynamics_numerical(
            list_joint_angles,
            list_joint_velocities,
            [0.0] * i_num_joints,
            list_external_force_at_tool=np.zeros(6),
        )
        f_c_plus_g_np: np.ndarray = np.array(list_coriolis_gravity_torques)
        f_c_term: np.ndarray = f_c_plus_g_np - f_g_vec

        return mat_m, f_c_term, f_g_vec

    def forward_dynamics_numerical(
        self,
        list_joint_angles: list[float],
        list_joint_velocities: list[float],
        list_joint_torques: list[float],
        list_external_force_at_tool: list[float] = None,
    ) -> list[float]:
        """
        Calculates joint accelerations (q_ddot) given joint positions, velocities, and applied torques.
        It uses the dynamic equation: M(q) * q_ddot + C(q, q_dot) + G(q) = tau
        Solving for q_ddot: q_ddot = M_inv * (tau - C - G)

        @param list_joint_angles: Current joint positions in radians.
        @param list_joint_velocities: Current joint velocities in rad/s.
        @param list_joint_torques: Torques applied to the joints in Nm.
        @param list_external_force_at_tool: Optional external force/torque at the end-effector.
                                             Note: For this simplified implementation, external forces
                                             are assumed to be either handled outside this method
                                             or are zero.

        @return list[float]: Resulting joint accelerations in rad/s^2.
        """
        i_num_joints: int = self.robot.i_num_joints

        # Get M, C, G for the current state
        mat_m, f_c_term, f_g_term = self.get_mass_matrix_and_coriolis_gravity(
            list_joint_angles, list_joint_velocities
        )

        f_tau_applied: np.ndarray = np.array(list_joint_torques)

        # Solve for q_ddot: q_ddot = M_inv * (tau_applied - C_term - G_term)
        try:
            mat_m_inv: np.ndarray = np.linalg.inv(mat_m)
        except np.linalg.LinAlgError:
            print(
                "Warning: Mass matrix is singular or ill-conditioned. Returning zero accelerations."
            )
            return [0.0] * i_num_joints

        f_q_ddot: np.ndarray = mat_m_inv @ (f_tau_applied - f_c_term - f_g_term)

        return f_q_ddot.tolist()
