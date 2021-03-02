import numpy as np

from python_utils import mathu

from scipy.spatial.transform import Rotation

from ilc_models.base import g, g3
from ilc_models.quad3d import Quad3D

class Quad3DTV(Quad3D):
  """
    state is (pos, vel, rpy, angular velocity)
    control is (u, angular acceleration)

    Uses the thrust vector rotational error metric

    Here we us RPY in the state but the ILC matrices,
    including the feedback response, are calculated in
    terms of the body z-axis instead of Euler angles
    This simplifies the gradient considerably.

    This is okay since the ILC matrices only interact with
    the true state using position, not the attitude.
  """
  K_pos = np.array((
    (5.47368, 0, 0, 3.15789, 0, 0),
    (0, 5.47368, 0, 0, 3.15789, 0),
    #(0, 0, 10, 0, 0, 6)
    (0, 0, 5.47368, 0, 0, 3.15789)
  ))

  K_att = np.array((
    (190, 0, 0, 25, 0, 0),
    (0, 190, 0, 0, 25, 0),
    (0, 0, 30, 0, 0, 10)
  ))

  control_normalization = np.array((0.1, 1e-2, 1e-2, 1e-2))

  def get_feedback_response(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    Z = slice(6, 9)
    RPY = slice(6, 9)
    OM = slice(9, 12)
    U = slice(0, 1)
    AA = slice(1, 4)

    pos = state[X]
    vel = state[V]
    rpy = state[RPY]
    angvel = state[OM]

    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))
    skew_z = mathu.skew_matrix(z)

    rot_inv = rot.inv().as_matrix()

    pos_vel = state[:6]
    accel_des = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + g3
    anorm = np.linalg.norm(accel_des)

    K_x = np.zeros((self.n_control, self.n_state))
    K_u = np.zeros((self.n_control, self.n_control))

    dadx = -self.K_pos[:, X]
    dadv = -self.K_pos[:, V]

    duda = self.U.duda(accel_des, z)
    dudz = self.U.dudz(accel_des, z)

    dudx = duda.dot(dadx)
    dudv = duda.dot(dadv)

    K_x[U, X] = dudx
    K_x[U, V] = dudv
    K_x[U, Z] = dudz

    z_des = accel_des / anorm

    dzda = (1.0 / anorm) * (np.eye(3) - np.outer(z_des, z_des))
    dzdx = dzda.dot(dadx)
    dzdv = dzda.dot(dadv)

    K_x[AA, X] = self.K_att[:, :3].dot(rot_inv).dot(skew_z.dot(dzdx))
    K_x[AA, V] = self.K_att[:, :3].dot(rot_inv).dot(skew_z.dot(dzdv))

    # TODO Fix this.
    K_x[AA, Z] = -self.K_att[:, :3].dot(rot_inv).dot(mathu.skew_matrix(z_des))
    K_x[AA, OM] = -self.K_att[:, 3:6]

    K_u[U, U] = 1
    K_u[AA, AA] = np.eye(3)

    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    RPY = slice(6, 9)
    Z = slice(6, 9)
    OM = slice(9, 12)
    U = slice(0, 1)
    AA = slice(1, 4)

    rpy = state[RPY]
    angvel = state[OM]
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))

    ang_world = rot.apply(angvel)

    if not self.use_feedback:
      u = control[U][0]
    else:
      pos_vel = state[:6]
      accel_des = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + self.g_vec
      u = self.U.u(accel_des, z)

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(3)
    A[X, V] = dt * np.eye(3)

    A[V, V] = np.eye(3)
    A[V, Z] = u * dt * np.eye(3)

    A[Z, Z] = np.eye(3) + dt * mathu.skew_matrix(ang_world)
    #A[Z, OM] = -dt * mathu.skew_matrix(z).dot(rot.as_matrix())
    A[Z, OM] = -dt * rot.as_matrix().dot(mathu.skew_matrix(np.array((0, 0, 1))))

    A[OM, OM] = np.eye(3)

    B[V, U] = dt * np.array([z]).T
    B[OM, AA] = dt * np.eye(3)

    C[X, X] = np.eye(3)

    if self.use_feedback:
      K_x, K_u = self.get_feedback_response(state, control, dt)

      A += B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, angaccel_des, u_ilc, **kwargs):
    pos_vel = x[:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))

    if self.model_drag:
      drag_dist_control = self.drag_dist
    else:
      drag_dist_control = 0

    # Position Control
    pos_vel_error = pos_vel - np.hstack((pos_des, vel_des))
    accel_des = -self.K_pos.dot(pos_vel_error) + acc_des + self.g_vec + drag_dist_control * pos_vel[3:]
    z_des = accel_des / np.linalg.norm(accel_des)

    u_accel = self.U.u(accel_des, z)

    # Attitude Control
    rot_error_w = np.cross(z_des, z)
    # Move rot_error to the body frame.
    rot_error_b = rot.inv().apply(rot_error_w)
    angvel_error = ang_vel - angvel_des
    rot_angvel = np.hstack((rot_error_b, angvel_error))
    u_ang_accel = -self.K_att.dot(rot_angvel) + angaccel_des

    return np.hstack((u_accel + u_ilc[0], u_ang_accel + u_ilc[1:]))
