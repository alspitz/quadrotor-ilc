import numpy as np

from scipy.spatial.transform import Rotation

from ilc_models.base import g, g3
from ilc_models.quad3dflv import Quad3DFLV

K1xy = 1040
K2xy = 600
K3xy = 190
K4xy = 25

#K1z = 1900
#K2z = 1140
K1z = 1040
K2z = 600
K3z = 190
K4z = 25

class Quad3DFLS(Quad3DFLV):
  """ This system is augmented with the two virtual control integrators
      used in dynamic extension to delay the appearance of the thrust
      control input u
      ILC corrects snap resulting in a linear system
  """
  n_state = 3 * 4
  n_control = 3
  n_out = 3

  state_labels = [
    "Position X",
    "Position Y",
    "Position Z",
    "Velocity X",
    "Velocity Y",
    "Velocity Z",
    "Roll",
    "Pitch",
    "Yaw",
    "Roll Velocity",
    "Pitch Velocity",
    "Yaw Velocity",
    "Thrust",
    "Thrust Velocity"
  ]

  control_labels = [ "Snap X", "Snap Y", "Snap Z" ]
  control_normalization = np.array((1e-2 / g, 1e-2 / g, 1e-2 / g))
  #control_normalization = np.array((0.1, 1e-2, 1e-2, 1e-2))
  constant_ilc_mats = True

  def __init__(self, *args, **kwargs):
    super(Quad3DFLS, self).__init__(*args, **kwargs)

    self.k1 = K1xy * np.eye(self.n_control)
    self.k2 = K2xy * np.eye(self.n_control)
    self.k3 = K3xy * np.eye(self.n_control)
    self.k4 = K4xy * np.eye(self.n_control)

    self.k1[2, 2] = K1z
    self.k2[2, 2] = K2z
    self.k3[2, 2] = K3z
    self.k4[2, 2] = K4z

  def get_feedback_response(self, state, control, dt):
    dims = 3
    devs = 4
    n_st = dims * devs

    ks = [self.k1, self.k2, self.k3, self.k4]
    K_x = np.zeros((dims, n_st))
    for i in range(devs):
      K_x[:, i * dims : (i + 1) * dims] = -ks[i]

    K_u = np.eye(dims)
    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    dims = 3
    devs = 4
    n_st = dims * devs

    A = np.eye(n_st)
    for i in range(devs - 1):
      A[i * dims : (i + 1) * dims, (i + 1) * dims:(i + 2) * dims] = dt * np.eye(dims)

    B = np.zeros((n_st, dims))
    B[(devs - 1) * dims : devs * dims, :] = dt * np.eye(dims)

    C = np.zeros((self.n_out, self.n_state))
    C[0:dims, 0:dims] = np.eye(dims)

    D = np.zeros((self.n_out, self.n_control))

    if self.use_feedback:
      K_x, K_u = self.get_feedback_response(state, control, dt)
      A += B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedback(self, x, dt, pos_des, vel_des, acc_des, jerk_des, snap_des, u_ilc, integrate=True, **kwargs):
    pos = x[:3]
    vel = x[3:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    self.zs.append(np.array((self.int_u, self.int_udot)))

    rot = Rotation.from_euler('ZYX', rpy[::-1])
    self.z_b_act = z_b_act = rot.apply(np.array((0, 0, 1)))

    ang_world = rot.apply(ang_vel)
    z_b_dot_act = np.cross(ang_world, z_b_act)

    u = self.int_u
    udot = self.int_udot

    if self.model_drag:
      drag_dist_control = self.drag_dist
    else:
      drag_dist_control = 0

    start_acc = u * z_b_act - g3 - drag_dist_control * vel
    start_jerk = u * z_b_dot_act + udot * z_b_act - drag_dist_control * start_acc

    self.z = z_b_act
    self.acc = start_acc

    pos_err = pos - pos_des
    vel_err = vel - vel_des
    acc_err = start_acc - acc_des
    jerk_err = start_jerk - jerk_des

    # Linear controller
    snap = -self.k1.dot(pos_err) - self.k2.dot(vel_err) - self.k3.dot(acc_err) - self.k4.dot(jerk_err) + snap_des + u_ilc
    self.snap = snap

    v1 = snap.dot(z_b_act) + u * z_b_dot_act.dot(z_b_dot_act) + drag_dist_control * start_jerk.dot(z_b_act)
    self.v1 = v1

    # Here we don't include v1 * z_b_act, because when crossed with Z, this term is zero.
    z_ddot = (1 / u) * (snap - 2 * udot * z_b_dot_act + drag_dist_control * start_jerk)

    ang_acc_world = np.cross(z_b_act, z_ddot) - (ang_world.dot(z_b_act)) * np.cross(z_b_act, ang_world)
    ang_acc_body = rot.inv().apply(ang_acc_world)

    u_ret = self.int_u

    if integrate:
      self.int_u += self.int_udot * dt
      self.int_udot += v1 * dt

      self.int_u = np.clip(self.int_u, -10000, 10000)
      self.int_udot = np.clip(self.int_udot, -10000, 10000)

    control = np.hstack((u_ret, ang_acc_body))
    return control
