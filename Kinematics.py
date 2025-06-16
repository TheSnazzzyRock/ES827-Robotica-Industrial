## @file Kinematics.py
#  @brief Implements the Kinematics class for robot kinematic calculations.
#         Provides forward and inverse kinematics functionalities.
#  @author Grupo 1
#  @date June 7, 2025
#  @version 1.0
#
#  This file defines the Kinematics class, which uses the Denavit-Hartenberg (DH)
#  parameters from the Robot model to calculate the robot's end-effector pose
#  (forward kinematics) and joint angles for a desired pose (inverse kinematics).

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R_scipy
from sympy import Matrix, cos, pi, sin, symbols, pretty_print, diff


from Robot import Robot


class Kinematics:
    """
    Class for performing robot kinematic calculations (forward and inverse)
    for a robot defined by Denavit-Hartenberg (DH) parameters.
    """

    def __init__(self, robot_model: Robot):
        """
        Initializes the kinematics calculator with the robot model.

        @param robot_model: An instance of the Robot class containing the robot's parameters.
        """
        self.robot = robot_model
        # Pre-calculate the symbolic homogeneous transformation matrix once
        self._mat_t0_n_symbolic: Matrix = self._calculate_symbolic_fk()
        # Pre-calculate the symbolic Jacobian matrix once
        self._mat_jacobian_symbolic: Matrix = self._calculate_symbolic_jacobian()

    def _calculate_dh_matrix(
        self, f_a: symbols, f_alpha: symbols, f_d: symbols, f_theta: symbols
    ) -> Matrix:
        """
        Calculates the homogeneous transformation matrix for a single link
        using Denavit-Hartenberg (DH) parameters.

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
                    -cos(f_theta) * sin(f_alpha),
                    f_a * sin(f_theta),
                ],
                [0, sin(f_alpha), cos(f_alpha), f_d],
                [0, 0, 0, 1],
            ]
        )

    def _calculate_symbolic_fk(self) -> Matrix:
        """
        Calculates the total symbolic homogeneous transformation matrix (T0_N).
        This represents the forward kinematics from the base to the end-effector.

        @return sympy.Matrix: A 4x4 symbolic homogeneous transformation matrix,
                              as a function of joint symbols and DH parameters.
        """
        mat_t_total: Matrix = Matrix.eye(4)
        for params in self.robot.list_dh_params_sym:
            f_a, f_alpha, f_d, f_theta = params
            mat_t_link = self._calculate_dh_matrix(f_a, f_alpha, f_d, f_theta)
            mat_t_total = mat_t_total @ mat_t_link
        return mat_t_total

    @property
    def symbolic_fk_matrix(self) -> Matrix:
        """
        Returns the pre-calculated symbolic forward kinematics (T0_N) matrix.

        @return sympy.Matrix: The pre-calculated 4x4 symbolic homogeneous transformation matrix.
        """
        return self._mat_t0_n_symbolic

    def _calculate_symbolic_jacobian(self) -> Matrix:
        """
        Calculates the symbolic Geometric Jacobian matrix of the robot.
        This method is applicable for revolute (rotational) joints only.

        @return sympy.Matrix: A 6xN_joints symbolic Jacobian matrix.
        """
        mat_jacobian: Matrix = Matrix.zeros(6, self.robot.i_num_joints)

        # Position of the end-effector in the base frame
        pos_p0_n: Matrix = self.symbolic_fk_matrix[:3, 3]

        # Initialize transformation from base to previous joint frame
        mat_t_prev: Matrix = Matrix.eye(4)

        for j in range(self.robot.i_num_joints):
            # Get DH parameters for the current joint (j)
            params_j = self.robot.list_dh_params_sym[j]

            # Calculate transformation from base to current joint frame (T0_j)
            mat_t0_j = mat_t_prev @ self._calculate_dh_matrix(
                params_j[0], params_j[1], params_j[2], params_j[3]
            )
            # Z-axis of the previous joint frame (represents the joint's axis of rotation)
            vec_z_j_minus_1: Matrix = mat_t_prev[:3, 2]
            # Position of the origin of the previous joint frame
            pos_p_j_minus_1: Matrix = mat_t_prev[:3, 3]

            # Linear component of the Jacobian (first 3 rows)
            # z_{j-1} x (p_{end-effector} - p_{j-1})
            vec_linear_component: Matrix = vec_z_j_minus_1.cross(
                pos_p0_n - pos_p_j_minus_1
            )

            # Angular component of the Jacobian (last 3 rows)
            # z_{j-1}
            vec_angular_component: Matrix = vec_z_j_minus_1

            # Assign columns to the Jacobian matrix
            for row in range(3):
                mat_jacobian[row, j] = vec_linear_component[row]
                mat_jacobian[row + 3, j] = vec_angular_component[row]

            # Update T_prev for the next iteration
            mat_t_prev = mat_t0_j

        return mat_jacobian

    @property
    def symbolic_jacobian_matrix(self) -> Matrix:
        """
        Returns the pre-calculated symbolic Jacobian matrix.

        @return sympy.Matrix: The pre-calculated 6xN_joints symbolic Jacobian matrix.
        """
        return self._mat_jacobian_symbolic

    def calculate_numerical_fk(self, list_joint_angles_rad: list[float]) -> Matrix:
        """
        Calculates the numerical forward kinematics for a given joint configuration.

        @param list_joint_angles_rad: A list of joint angles in radians.
        @return sympy.Matrix: A 4x4 numerical homogeneous transformation matrix.
        @raises ValueError: If the number of provided angles does not match the number of joints.
        """
        if len(list_joint_angles_rad) != self.robot.i_num_joints:
            raise ValueError(
                f"Expected {self.robot.i_num_joints} angles, but received {len(list_joint_angles_rad)}."
            )

        # Dictionary to substitute symbolic joint variables with numerical values
        dict_subs_q = {
            self.robot.list_joint_vars[i]: angle
            for i, angle in enumerate(list_joint_angles_rad)
        }

        # Dictionary to substitute symbolic DH parameters (a, alpha, d) with numerical values
        dict_subs_dh_params = {}
        for i in range(self.robot.i_num_joints):
            dict_subs_dh_params[self.robot.list_dh_params_sym[i][0]] = (
                self.robot.list_numerical_a[i]
            )
            dict_subs_dh_params[self.robot.list_dh_params_sym[i][1]] = (
                self.robot.list_numerical_alpha[i]
            )
            dict_subs_dh_params[self.robot.list_dh_params_sym[i][2]] = (
                self.robot.list_numerical_d[i]
            )

        # Combine all substitution dictionaries
        dict_all_subs = {**dict_subs_q, **dict_subs_dh_params}

        # Substitute all symbols in the pre-calculated symbolic matrix
        mat_t_numerical: Matrix = self.symbolic_fk_matrix.subs(dict_all_subs)
        return mat_t_numerical

    def calculate_numerical_jacobian(
        self, list_joint_angles_rad: list[float]
    ) -> np.ndarray:
        """
        Calculates the numerical Jacobian matrix for a given joint configuration.

        @param list_joint_angles_rad: A list of joint angles in radians.
        @return np.ndarray: A numerical Jacobian matrix (6xN_joints) as a NumPy array.
        @raises ValueError: If the number of provided angles does not match the number of joints.
        """
        if len(list_joint_angles_rad) != self.robot.i_num_joints:
            raise ValueError(
                f"Expected {self.robot.i_num_joints} angles, but received {len(list_joint_angles_rad)}."
            )

        # Dictionary to substitute symbolic joint variables with numerical values
        dict_subs_q = {
            self.robot.list_joint_vars[i]: angle
            for i, angle in enumerate(list_joint_angles_rad)
        }

        # Dictionary to substitute symbolic DH parameters (a, alpha, d) with numerical values
        dict_subs_dh_params = {}
        for i in range(self.robot.i_num_joints):
            dict_subs_dh_params[self.robot.list_dh_params_sym[i][0]] = (
                self.robot.list_numerical_a[i]
            )
            dict_subs_dh_params[self.robot.list_dh_params_sym[i][1]] = (
                self.robot.list_numerical_alpha[i]
            )
            dict_subs_dh_params[self.robot.list_dh_params_sym[i][2]] = (
                self.robot.list_numerical_d[i]
            )

        # Combine all substitution dictionaries
        dict_all_subs = {**dict_subs_q, **dict_subs_dh_params}

        # Substitute symbols in the symbolic Jacobian matrix
        jacobian_numerical_sympy: Matrix = self.symbolic_jacobian_matrix.subs(
            dict_all_subs
        )

        # Convert to a NumPy array for efficient numerical operations
        jacobian_numerical_np: np.ndarray = np.array(jacobian_numerical_sympy).astype(
            np.float64
        )
        return jacobian_numerical_np

    def calculate_inverse_kinematics(
        self, mat_target_pose: Matrix, list_initial_guess: list[float] = None
    ) -> list[float]:
        """
        Calculates the inverse kinematics numerically for a target pose using
        scipy.optimize.minimize. This is a robust approach for robots with
        multiple degrees of freedom.

        @param mat_target_pose: The desired 4x4 end-effector pose matrix.
        @param list_initial_guess: An optional list of initial joint angles in radians.
                                    If None, uses the home (zero) position as the initial guess.
        @return list[float]: The joint angles in radians that achieve the target pose,
                             or the initial guess if the optimization does not converge.
        """

        if list_initial_guess is None:
            list_initial_guess = [0.0] * self.robot.i_num_joints

        # Convert target pose matrix to NumPy position vector and quaternion
        f_target_pos_np: np.ndarray = (
            np.array(mat_target_pose[:3, 3]).astype(np.float64).flatten()
        )
        mat_target_rot_np: np.ndarray = np.array(mat_target_pose[:3, :3]).astype(
            np.float64
        )
        # Convert rotation matrix to quaternion (x, y, z, w)
        f_target_quat: np.ndarray = R_scipy.from_matrix(mat_target_rot_np).as_quat()

        def cost_function(f_q_angles: np.ndarray) -> float:
            """
            Cost function for inverse kinematics optimization.
            Minimizes the difference between the current pose and the target pose.

            @param f_q_angles: Current joint angles as a NumPy array.
            @return float: The total error (cost) value.
            """
            mat_current_t_sympy: Matrix = self.calculate_numerical_fk(
                f_q_angles.tolist()
            )
            mat_current_t_np: np.ndarray = np.array(mat_current_t_sympy).astype(
                np.float64
            )

            f_current_pos_np: np.ndarray = mat_current_t_np[:3, 3]
            mat_current_rot_np: np.ndarray = mat_current_t_np[:3, :3]
            f_current_quat: np.ndarray = R_scipy.from_matrix(
                mat_current_rot_np
            ).as_quat()

            # Position error (Euclidean norm of the position vector difference)
            f_pos_error: float = np.linalg.norm(f_target_pos_np - f_current_pos_np)

            # Orientation error
            f_dot_product: float = np.dot(f_target_quat, f_current_quat)
            f_quat_error: float = 1.0 - np.abs(f_dot_product)

            # Combined position and orientation error (with weighting)
            f_total_error: float = f_pos_error + 10 * f_quat_error
            return f_total_error

        # Define joint limits
        list_joint_bounds: list[tuple[float, float]] = [
            (-2.9, 2.9),   
            (-1.8, 2.1),   
            (-2.8, 2.3),  
            (-3.0, 3.0),  
            (-2.0, 2.0),   
            (-6.28, 6.28)  
        ]

        # Execute the optimization (Sequential Least Squares Programming)
        result = minimize(
            cost_function,
            list_initial_guess,
            method="SLSQP",
            bounds=list_joint_bounds,
            tol=1e-6,
        )

        if result.success:
            print(
                f"Inverse kinematics converged successfully in {result.nit} iterations."
            )
            print(f"Final error (cost): {result.fun:.6e}")
            return result.x.tolist()
        else:
            print(f"Inverse kinematics did not converge: {result.message}")
            return list_initial_guess

    def calculate_inverse_kinematics_jacobian_pseudo_inverse(
        self,
        mat_target_pose: Matrix,
        list_initial_guess: list[float] = None,
        i_max_iterations: int = 100,
        f_tolerance: float = 1e-6,
        f_learning_rate: float = 0.1,
    ) -> list[float]:
        """
        Implements numerical inverse kinematics using the Jacobian pseudo-inverse (Damped Least Squares - DLS).
        This method is kept for reference/comparison with the optimization approach.

        @param mat_target_pose: The desired 4x4 end-effector pose matrix.
        @param list_initial_guess: An optional list of initial joint angles in radians.
                                    If None, uses the home (zero) position.
        @param i_max_iterations: Maximum number of algorithm iterations.
        @param f_tolerance: Minimum error for considering the solution converged.
        @param f_learning_rate: Scaling factor for the joint angle update step.
        @return list[float]: The joint angles in radians that achieve the target pose,
                             or the last estimate if it does not converge.
        """

        if list_initial_guess is None:
            f_current_joint_angles: np.ndarray = np.array(
                [0.0] * self.robot.i_num_joints
            )
        else:
            f_current_joint_angles = np.array(list_initial_guess).astype(np.float64)

        f_target_position: np.ndarray = (
            np.array(mat_target_pose[:3, 3]).astype(np.float64).flatten()
        )
        mat_target_orientation: np.ndarray = np.array(mat_target_pose[:3, :3]).astype(
            np.float64
        )

        for iteration in range(i_max_iterations):
            mat_current_t_sympy: Matrix = self.calculate_numerical_fk(
                f_current_joint_angles.tolist()
            )
            mat_current_t_np: np.ndarray = np.array(mat_current_t_sympy).astype(
                np.float64
            )

            f_current_position: np.ndarray = mat_current_t_np[:3, 3]
            mat_current_orientation: np.ndarray = mat_current_t_np[:3, :3]

            # Position error (3x1 vector)
            f_error_position: np.ndarray = f_target_position - f_current_position

            # Orientation error (using rotational vector approximation)
            mat_r_delta: np.ndarray = mat_target_orientation @ mat_current_orientation.T
            mat_error_orientation_skew: np.ndarray = mat_r_delta - np.eye(3)
            f_error_orientation: np.ndarray = (
                np.array(
                    [
                        mat_error_orientation_skew[2, 1],
                        mat_error_orientation_skew[0, 2],
                        mat_error_orientation_skew[1, 0],
                    ]
                )
                * 0.5
            )

            f_error_vector: np.ndarray = np.concatenate(
                (f_error_position, f_error_orientation)
            )

            if np.linalg.norm(f_error_vector) < f_tolerance:
                print(
                    f"Converged in {iteration} iterations. Final error: {np.linalg.norm(f_error_vector):.6e}"
                )
                return f_current_joint_angles.tolist()

            mat_jacobian_numerical: np.ndarray = self.calculate_numerical_jacobian(
                f_current_joint_angles.tolist()
            )

            # Calculate the Jacobian Pseudo-inverse (using Damped Least Squares - DLS)
            f_damping_factor: float = 0.001
            mat_j_t_j: np.ndarray = mat_jacobian_numerical.T @ mat_jacobian_numerical
            mat_j_t_j_damped: np.ndarray = mat_j_t_j + f_damping_factor * np.eye(
                mat_j_t_j.shape[0]
            )
            mat_jacobian_pseudo_inverse: np.ndarray = (
                np.linalg.inv(mat_j_t_j_damped) @ mat_jacobian_numerical.T
            )

            # Update joint angles
            f_delta_q: np.ndarray = f_learning_rate * (
                mat_jacobian_pseudo_inverse @ f_error_vector
            )
            f_current_joint_angles += f_delta_q

        print(
            f"Warning: Did not converge after {i_max_iterations} iterations. Final error: {np.linalg.norm(f_error_vector):.6e}"
        )
        return f_current_joint_angles.tolist()
