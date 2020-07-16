import numpy as np

from scipy.spatial.transform import Rotation

from ilc_models.base import g, g3
from ilc_models.quad3dfls import Quad3DFLS

K1 = 1040
K2 = 600
K3 = 190
K4 = 25

class Quad3DFLTD(Quad3DFLS):
  """ This system is augmented with the two virtual control integrators
      used in dynamic extension to delay the appearance of the thrust
      control input u.

      However, this system also models the thrust using a first order delay
      time constant. """

  n_state = 12
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
    "Thrust Commanded", # U
    "Thrust Estimated", # C
  ]

  control_labels = [
    "Thrust Velocity",
    "Roll Accel",
    "Pitch Accel",
    "Yaw Accel"
  ]

  control_normalization = np.array((1e-2 / g, 1e-2 / g, 1e-2 / g))

  def __init__(self, *args, **kwargs):
    super(Quad3DFLTD, self).__init__(*args, **kwargs)
    self.T_c = kwargs['delay_timeconstant_control']

  def feedback(self, x, dt, pos_des, vel_des, acc_des, jerk_des, snap_des, u_ilc, integrate=True, **kwargs):
    pos = x[:3]
    vel = x[3:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    self.zs.append(np.array((self.int_u, self.int_c)))

    rot = Rotation.from_euler('ZYX', rpy[::-1])
    self.z_b_act = z_b_act = rot.apply(np.array((0, 0, 1)))

    ang_world = rot.apply(ang_vel)
    z_b_dot_act = np.cross(ang_world, z_b_act)

    # thrust delay
    cdot = -self.T_c * (self.int_c - self.int_u)

    u = self.int_c
    udot = cdot

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

    k1 = K1 * np.eye(3) / self.duration ** 4
    k2 = K2 * np.eye(3) / self.duration ** 3
    k3 = K3 * np.eye(3) / self.duration ** 2
    k4 = K4 * np.eye(3) / self.duration ** 1

    # Linear controller
    snap = -k1.dot(pos_err) - k2.dot(vel_err) - k3.dot(acc_err) - k4.dot(jerk_err) + snap_des + u_ilc
    self.snap = snap

    # thrust delay
    self.int_udot = (snap.dot(z_b_act) + u * z_b_dot_act.dot(z_b_dot_act)) / self.T_c + cdot# + u_ilc[0]

    udot = self.int_udot

    # Here we don't include v1 * z_b_act, because when crossed with Z, this term is zero.
    z_ddot = (1 / u) * (snap - 2 * udot * z_b_dot_act + drag_dist_control * start_jerk)

    ang_acc_world = np.cross(z_b_act, z_ddot) - (ang_world.dot(z_b_act)) * np.cross(z_b_act, ang_world)
    ang_acc_body = rot.inv().apply(ang_acc_world)

    u_ret = self.int_u

    if integrate:
      self.int_u += self.int_udot * dt
      self.int_c += cdot * dt

      self.int_u = np.clip(self.int_u, -10000, 10000)
      self.int_c = np.clip(self.int_c, -10000, 10000)

    control = np.hstack((u_ret, ang_acc_body))
    return control
