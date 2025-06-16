## @file Dynamics.py
#  @brief Implements the Dynamics class using the Pinocchio library.
#  @author Grupo 1
#  @date June 16, 2025 (Final Delivery Version)
#
#  This class acts as a wrapper for the Pinocchio library, providing a clean
#  interface for robot dynamics calculations (Inverse and Forward Dynamics).
#  It builds the internal physics model from the Robot data container.

import numpy as np
import pinocchio as pin
from Robot import Robot
from Kinematics import Kinematics

class Dynamics:
    """
    Performs robot dynamics calculations using the Pinocchio library.
    This class builds a Pinocchio model from the Robot parameters and provides
    methods to compute M, C, G and forward dynamics.
    """
    def __init__(self, obj_robot_model: Robot, obj_kinematics_calculator: Kinematics):
        """
        Initializes the dynamics calculator.
        @param obj_robot_model: Instance of the Robot class.
        @param obj_kinematics_calculator: Instance of the Kinematics class.
        """
        self.robot_model = obj_robot_model
        self.kinematics = obj_kinematics_calculator
        self.pin_model = self._build_pinocchio_model()
        self.pin_data = self.pin_model.createData()
        self.pin_model.gravity.linear = -np.array(self.robot_model.mat_gravity_vector).astype(np.float64).flatten()

    def _build_pinocchio_model(self):
        """
        Builds the Pinocchio model from the parameters in the Robot class.
        @return: A configured Pinocchio model object.
        """
        model = pin.Model()
        parent_id = 0
        
        a, alpha, d = self.robot_model.list_numerical_a, self.robot_model.list_numerical_alpha, self.robot_model.list_numerical_d
        masses, com_vectors, inertia_tensors = self.robot_model.list_numerical_masses, self.robot_model.list_numerical_com_vectors, self.robot_model.list_numerical_inertia_tensors

        # DH Standard: T_i^{i-1} = Rot(z, q_i) * Trans(z, d_i) * Trans(x, a_i) * Rot(x, alpha_i)
        for i in range(self.robot_model.i_num_joints):
            # The JointModelRZ handles Rot(z, q_i), so the placement is the fixed part.
            T_trans_z = pin.SE3(np.eye(3), np.array([0, 0, d[i]]))
            T_trans_x = pin.SE3(np.eye(3), np.array([a[i], 0, 0]))
            T_rot_x = pin.SE3(pin.utils.rotate('x', alpha[i]), np.zeros(3))
            
            joint_placement = T_trans_z * T_trans_x * T_rot_x
            
            joint_id = model.addJoint(parent_id, pin.JointModelRZ(), joint_placement, f'joint_{i+1}')
            
            inertia = pin.Inertia(masses[i], com_vectors[i], inertia_tensors[i])
            model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
            parent_id = joint_id
            
        return model

    def get_mass_matrix_and_coriolis_gravity(self, list_q: list, list_qd: list):
        """
        Calculates the Mass Matrix (M), Coriolis Vector (C), and Gravity Vector (G).
        @param list_q: Current joint positions (list of floats).
        @param list_qd: Current joint velocities (list of floats).
        @return: A tuple (M, C, G) as NumPy arrays.
        """
        q, qd = np.array(list_q), np.array(list_qd)
        
        M = pin.crba(self.pin_model, self.pin_data, q)
        C_G = pin.rnea(self.pin_model, self.pin_data, q, qd, np.zeros(self.pin_model.nv))
        G = pin.rnea(self.pin_model, self.pin_data, q, np.zeros(self.pin_model.nv), np.zeros(self.pin_model.nv))
        C = C_G - G
        return M, C, G

    def forward_dynamics_numerical(self, list_q: list, list_qd: list, list_tau: list):
        """
        Calculates joint accelerations from applied torques (Forward Dynamics).
        @param list_q: Current joint positions.
        @param list_qd: Current joint velocities.
        @param list_tau: Applied joint torques.
        @return: A list of joint accelerations.
        """
        q, qd, tau = np.array(list_q), np.array(list_qd), np.array(list_tau)
        qdd = pin.aba(self.pin_model, self.pin_data, q, qd, tau)
        return qdd.tolist()