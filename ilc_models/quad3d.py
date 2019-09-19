import numpy as np

import math_utils

from scipy.spatial.transform import Rotation

from ilc_models.base import ILCBase, g, g3
from lqr_gain_match.match_full_state import accel_to_euler_rpy

class Quad3D(ILCBase):
  """
    state is (pos, vel, z axis, angular velocity)
    control is (u, angular acceleration)

    #z_next = z + dt * [angvel]_x[:, 3]
    #angvel_next = omega + dt * angular acceleration

    #obs = accel = u * z axis + g
  """
  n_state = 12
  n_control = 4
  n_out = 3

  K_pos = np.array((
    (7.0, 0, 0, 4.0, 0, 0),
    (0, 7.0, 0, 0, 4.0, 0),
    (0, 0, 7.0, 0, 0, 4.0)
  ))

  K_att = np.array((
    (120, 0, 0, 16, 0, 0),
    (0, 120, 0, 0, 16, 0),
    (0, 0, 60, 0, 0, 12)
  ))

  control_normalization = np.array((1e-1, 1e-2, 1e-2, 1e-2))

  use_snap = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.model_drag = kwargs['model_drag']
    self.drag_dist = kwargs['drag_dist']
    self.thrust_dist = kwargs['thrust_dist']

  def get_ABCD(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    Z = slice(6, 9)
    OM = slice(9, 12)
    U = slice(0, 1)
    AA = slice(1, 4)

    u = control[U][0]
    rpy = state[Z]
    angvel = state[OM]
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(3)
    A[V, V] = np.eye(3)
    A[X, V] = dt * np.eye(3)

    A[Z, Z] = np.eye(3)
    A[OM, OM] = np.eye(3)

    A[V, Z] = u * dt * np.eye(3)
    # TODO Need to fix this because om is in the body frame!
    A[Z, OM] = -dt * math_utils.skew_matrix(z)

    B[V, U] = dt * np.array([z]).T
    B[OM, AA] = dt * np.eye(3)

    # Trying 2D mats.......
    #ct = np.cos(rpy[0])
    #st = np.sin(rpy[0])
    #A[V, 6:7] = u * dt * np.array(((0, -ct, -st),)).T
    #A[6, 9] = dt
    #########################33

    C[X, X] = np.eye(3)

    if self.use_feedback:
      K_x = np.zeros((self.n_control, self.n_state))
      K_u = np.zeros((self.n_control, self.n_control))

      pos_vel = state[:6]
      a = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + g3
      adota = a.T.dot(a)
      adir = a / np.sqrt(adota)

      K_x[U, X] = adir.dot(-self.K_pos[:, X])
      K_x[U, V] = adir.dot(-self.K_pos[:, V])

      #print(np.cos(rpy[0]) * np.sin(rpy[1]), -np.sin(rpy[0]), np.cos(rpy[1]) * np.cos(rpy[0]))
      #print(z.T)
      #input()

      deulerdz = np.zeros((3, 3))
      a1 = 1 / np.sqrt(1 - z[1] ** 2)
      deulerdz[0, 1] = -a1
      deulerdz[1, 0] = 1 / np.sqrt(1 - (z[0] * a1) ** 2)
      deulerdz[1, 1] = z[0] * z[1] / (np.sqrt(1 - (z[0] * a1) ** 2) * ((1 - z[1]**2) ** (3/2)))

      K_x[AA, Z] = -self.K_att[:3, :3].dot(deulerdz)
      K_x[AA, OM] = -self.K_att[:3, 3:6]

      K_u[U, U] = 1
      K_u[AA, AA] = np.eye(3)

      A = A + B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedforward(self, pos, vel, acc, jerk, snap):
    acc_vec = acc + g3
    u = np.linalg.norm(acc_vec)

    if u < 1e-3:
      print("WARNING: acc norm too low!")

    z_b      = (1.0 / u) * acc_vec
    z_b_dot  = (1.0 / u) * (jerk - z_b.dot(jerk) * z_b)

    ang_vel = np.cross(z_b, z_b_dot)

    z_b_ddot = (1.0 / u) * (snap - (snap.dot(z_b) + jerk.dot(z_b_dot)) * z_b - 2 * jerk.dot(z_b) * z_b_dot) - np.cross(ang_vel, z_b_dot)

    y_b = np.cross(z_b, np.array((1, 0, 0)))
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)

    dp = z_b_ddot.dot(-y_b)
    dq = z_b_ddot.dot( x_b)

    ang_acc_body = np.array((dp, dq, 0))

    ang_vel_body = np.array((ang_vel.dot(x_b), ang_vel.dot(y_b), ang_vel.dot(z_b)))

    # TODO HMM XXX
    if self.use_feedback:
      u = g
      if self.use_snap:
        ang_acc_body *= 0

    state = np.hstack((pos, vel, z_b, ang_vel_body))
    control = np.hstack((u, ang_acc_body))

    return state, control

  def simulate(self, t_end, fun, dt):
    pos = np.zeros(3)
    vel = np.zeros(3)
    rpy = np.zeros(3)
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    quat = rot.as_quat()
    ang_body = np.zeros(3)
    ang_world = rot.apply(ang_body)

    x = np.zeros(12)
    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      u = u_out[0]
      ang_accel_body = np.array((u_out[1], u_out[2], u_out[3]))
      ang_accel_world = rot.apply(ang_accel_body)

      if u < 0:
        u = 0
        print("WARNING: THRUST IS NEGATIVE")

      ang_accel_limit = 500
      if np.any(np.abs(ang_accel_world) > ang_accel_limit):
        print("WARNING: Ang accel is very high!")
        ang_accel_world = np.clip(ang_accel_world, -ang_accel_limit, ang_accel_limit)


      acc = self.thrust_dist * u * rot.apply(np.array((0, 0, 1))) - g3 - self.drag_dist * vel

      pos += vel * dt
      vel += acc * dt

      quat_wfirst = np.array((quat[3], quat[0], quat[1], quat[2]))

      quat_deriv = math_utils.quat_mult(math_utils.vector_quat(ang_world), quat_wfirst) / 2.0
      quat_wfirst += quat_deriv * dt
      quat_wfirst /= np.linalg.norm(quat_wfirst)

      ang_world += ang_accel_world * dt
      quat = np.array((quat_wfirst[1], quat_wfirst[2], quat_wfirst[3], quat_wfirst[0]))
      rot = Rotation.from_quat(quat)
      rpy = rot.as_euler('ZYX')[::-1]
      ang_body = rot.inv().apply(ang_world)

      x = np.hstack((pos.copy(), vel.copy(), rpy.copy(), ang_body.copy()))

      xs.append(x)

    return np.array(xs)

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, u_ff, angaccel_des, **kwargs):
    pos_vel = x[:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    pos_vel_error = pos_vel - np.hstack((pos_des, vel_des))

    # Position Control
    accel_des = -self.K_pos.dot(pos_vel_error) + acc_des

    # Reference Conversion
    euler_des = accel_to_euler_rpy(accel_des, g)
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    #u_accel = rot.inv().apply(accel_des)[2]
    u_accel = np.linalg.norm(accel_des + g3)

    # Attitude Control
    euler_error = rpy - euler_des
    angvel_error = ang_vel - angvel_des
    euler_angvel = np.hstack((euler_error, angvel_error))
    u_ang_accel = -self.K_att.dot(euler_angvel) + angaccel_des

    return np.hstack((u_accel + u_ff - g, u_ang_accel))