import numpy as np

import math_utils

from scipy.spatial.transform import Rotation

from ilc_models.base import g, g3
from ilc_models.quad3d import Quad3D

class Quad3DFL(Quad3D):
  duration = 1.0
  int_u = g
  int_udot = 0

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.dt = kwargs['dt']

  def reset(self):
    self.int_u = g
    self.int_dot = 0
    self.zs = []

  def get_feedback_response(self, state, pos_des, vel_des, acc_des, jerk_des, snap_des):
    X = slice(0, 3)
    V = slice(3, 6)
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

    skew_z = math_utils.skew_matrix(z)

    skew_angvel_w = math_utils.skew_matrix(rot.apply(angvel))
    z_dot = skew_angvel_w.dot(z)

    roll, pitch, yaw = rpy

    dzdrpy = np.array((
      (np.sin(yaw) * np.cos(roll) - np.sin(roll) * np.cos(yaw) * np.sin(pitch), np.cos(roll) * np.cos(yaw) * np.cos(pitch), np.sin(roll) * np.cos(yaw) - np.cos(roll) * np.sin(yaw) * np.sin(pitch)),
      (-np.sin(roll) * np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.cos(roll), np.cos(roll) * np.sin(yaw) * np.cos(pitch), np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll)),
      (-np.cos(pitch) * np.sin(roll), -np.sin(pitch) * np.cos(roll), 0)
    ))

    dzdotdrpy = skew_angvel_w.dot(dzdrpy)

    dzdotdang = -skew_z.dot(rot.as_dcm())

    u, udot = self.int_u, self.int_udot

    k1 = 840 * np.eye(3) / self.duration ** 4
    k2 = 480 * np.eye(3) / self.duration ** 3
    k3 = 120 * np.eye(3) / self.duration ** 2
    k4 = 16 * np.eye(3) / self.duration ** 1

    start_acc = u * z - g3
    start_jerk = u * z_dot + udot * z

    pos_err = pos - pos_des
    vel_err = vel - vel_des
    acc_err = start_acc - acc_des
    jerk_err = start_jerk - jerk_des

    djerkdrpy = u * dzdotdrpy + udot * dzdrpy
    daccdrpy = u * dzdrpy

    dsnapdrpy = -k3.dot(daccdrpy) - k4.dot(djerkdrpy)

    snap = -k1.dot(pos_err) - k2.dot(vel_err) - k3.dot(acc_err) - k4.dot(jerk_err) + snap_des
    dv1drpy = dsnapdrpy.T.dot(z) + snap.dot(dzdrpy) + 2 * u * z_dot.T.dot(dzdotdrpy)

    v1 = snap.dot(z) + u * z_dot.dot(z_dot)

    u_factor = 1.0 / u
    z_ddot = u_factor * (snap - v1 * z - 2 * udot * z_dot)

    dzddotdrpy = u_factor * ( dsnapdrpy - np.outer(dv1drpy, z) - v1 * dzdrpy - 2 * udot * dzdotdrpy)

    dalphadrpy = skew_z.dot(dzddotdrpy - skew_angvel_w.dot(dzdotdrpy)) - math_utils.skew_matrix(z_ddot - skew_angvel_w.dot(z_dot)).dot(dzdrpy)

    djerkdang = u * dzdotdang
    dsnapdang = -k4.dot(djerkdang)
    dv1dang = dsnapdang.T.dot(z) + 2 * u * z_dot.T.dot(dzdotdang)
    dzddotdang = u_factor * (dsnapdang - np.outer(dv1dang, z) - 2 * udot * dzdotdang)
    # TODO XXX This is missing the omega cross zdot term.
    dalphawdang = skew_z.dot(dzddotdang)
    dalphabdang = rot.inv().apply(dalphawdang)

    dsnapdpos = -k1
    dsnapdvel = -k2

    dv1dpos = dsnapdpos.dot(z)
    dv1dvel = dsnapdvel.dot(z)

    dzddotdpos = u_factor * (dsnapdpos - np.outer(dv1dpos, z))
    dzddotdvel = u_factor * (dsnapdvel - np.outer(dv1dvel, z))

    dalphadpos = skew_z.dot(dzddotdpos)
    dalphadvel = skew_z.dot(dzddotdvel)

    fb_resp = np.zeros((self.n_control, self.n_state))
    fb_resp[AA, RPY] = dalphadrpy
    fb_resp[AA, OM] = dalphabdang
    fb_resp[AA, X] = dalphadpos
    fb_resp[AA, V] = dalphadvel
    return fb_resp

  def get_ABCD(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    RPY = slice(6, 9)
    OM = slice(9, 12)
    U = slice(0, 1)
    AA = slice(1, 4)

    pos = state[X]
    vel = state[V]

    ind = self.iter - 1 if self.iter >= len(self.zs) - 1 else self.iter
    u, udot = self.zs[ind]

    rpy = state[RPY]
    angvel = state[OM]
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))

    z_dot = math_utils.skew_matrix(rot.apply(angvel)).dot(z)

    roll, pitch, yaw = rpy

    dzdrpy = np.array((
      (np.sin(yaw) * np.cos(roll) - np.sin(roll) * np.cos(yaw) * np.sin(pitch), np.cos(roll) * np.cos(yaw) * np.cos(pitch), np.sin(roll) * np.cos(yaw) - np.cos(roll) * np.sin(yaw) * np.sin(pitch)),
      (-np.sin(roll) * np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.cos(roll), np.cos(roll) * np.sin(yaw) * np.cos(pitch), np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll)),
      (-np.cos(pitch) * np.sin(roll), -np.sin(pitch) * np.cos(roll), 0)
    ))

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(3)
    A[V, V] = np.eye(3)
    A[X, V] = dt * np.eye(3)

    A[RPY, RPY] = np.eye(3)
    A[OM, OM] = np.eye(3)

    #A[V, Z] = u * dt * np.eye(3)
    A[V, RPY] = u * dt * dzdrpy
    # TODO Need to fix this because om is in the body frame!
    #A[Z, OM] = -dt * math_utils.skew_matrix(z)

    # Below taken from Tal and Karaman 2018 - Accurate Tracking of ...
    A[RPY, OM] = dt * np.array((
      (1, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)),
      (0, np.cos(roll), -np.sin(roll)),
      (0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch))))

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

      oldu = self.int_u
      oldudot = self.int_udot

      self.int_u = u
      self.int_udot = udot

      K_x = self.get_feedback_response(state, self.pos_des, self.vel_des, self.acc_des, self.jerk_des, self.snap_des)

      self.int_u = oldu
      self.int_udot = oldudot

      K_u[U, U] = 1
      K_u[AA, AA] = np.eye(3)

      A = A + B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedback(self, x, pos_des, vel_des, acc_des, jerk_des, snap_des, u_ff, angaccel_ff, integrate=True):
    pos = x[:3]
    vel = x[3:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    self.zs.append(np.array((self.int_u, self.int_udot)))

    rot = Rotation.from_euler('ZYX', rpy[::-1])
    ang_vel = x[9:]
    rot_m = rot.as_dcm()
    self.x_b_act = rot_m[:, 0]
    self.y_b_act = rot_m[:, 1]
    self.z_b_act = z_b_act = rot_m[:, 2]

    ang_vel_w = rot.apply(ang_vel)
    z_b_dot_act = np.cross(ang_vel_w, z_b_act)

    u = self.int_u
    udot = self.int_udot

    if self.model_drag:
      DRAG_DIST_CONTROL = self.drag_dist
    else:
      DRAG_DIST_CONTROL = 0

    start_acc = u * z_b_act - g3 - DRAG_DIST_CONTROL * vel
    start_jerk = u * z_b_dot_act + udot * z_b_act - DRAG_DIST_CONTROL * start_acc

    pos_err = pos - pos_des
    vel_err = vel - vel_des
    acc_err = start_acc - acc_des
    jerk_err = start_jerk - jerk_des

    k1 = 840 * np.eye(3) / self.duration ** 4
    k2 = 480 * np.eye(3) / self.duration ** 3
    k3 = 120 * np.eye(3) / self.duration ** 2
    k4 = 16 * np.eye(3) / self.duration ** 1

    # Linear controller
    snap = -k1.dot(pos_err) - k2.dot(vel_err) - k3.dot(acc_err) - k4.dot(jerk_err) + snap_des

    v1 = snap.dot(z_b_act) + u * z_b_dot_act.dot(z_b_dot_act) + DRAG_DIST_CONTROL * start_jerk.dot(z_b_act)

    z_ddot = (1 / u) * (snap - v1 * z_b_act - 2 * udot * z_b_dot_act + DRAG_DIST_CONTROL * start_jerk)

    angaccel_cross_z = z_ddot - np.cross(ang_vel_w, z_b_dot_act)

    ang_acc_world = np.cross(z_b_act, angaccel_cross_z)
    ang_acc_body = rot.inv().apply(ang_acc_world) + angaccel_ff

    u_ret = self.int_u + u_ff

    if integrate:
      self.int_u += self.int_udot * self.dt + 0.5 * v1 * self.dt ** 2
      self.int_udot += v1 * self.dt

    control = np.hstack((u_ret, ang_acc_body))
    return control
