import numpy as np

from ilc_models.base import g, g2
from ilc_models.quad2ddedi import Quad2DDEDI

class Quad2DDEDIS(Quad2DDEDI):
  """
    state is (pos, vel, theta, omega, u, udot)
    control is (uddot, angular acceleration)
    ilc control is (snap)
  """

  control_labels = [ "Snap X", "Snap Z" ]
  control_normalization = np.array((1e-3, 1e-3))
  constant_ilc_mats = True

  def get_ABCD(self, state, control, dt):
    A = np.eye(8)
    A[0:2, 2:4] = dt * np.eye(2)
    A[2:4, 4:6] = dt * np.eye(2)
    A[4:6, 6:8] = dt * np.eye(2)

    B = np.zeros((8, 2))
    B[6:8, 0:2] = dt * np.eye(2)

    K_x = np.zeros((2, 8))
    K_x[:, 0:2] = -self.k1
    K_x[:, 2:4] = -self.k2
    K_x[:, 4:6] = -self.k3
    K_x[:, 6:8] = -self.k4

    A = A + B.dot(K_x)

    C = np.zeros((self.n_out, self.n_state))
    C[0:2, 0:2] = np.eye(2)

    D = np.zeros((self.n_out, self.n_control))

    return A, B, C, D

  def feedback(self, x, dt, pos_des, vel_des, acc_des, jerk_des, snap_des, u_ilc, integrate=True, **kwargs):
    pos = x[0:2]
    vel = x[2:4]
    theta = x[4]
    angvel = x[5]

    z = np.array((-np.sin(theta), np.cos(theta)))
    zdot = np.array((-np.cos(theta) * angvel, -np.sin(theta) * angvel))

    self.zs.append(np.array((self.int_u, self.int_udot)))

    acc = self.int_u * z - g2
    jerk = self.int_udot * z + self.int_u * zdot

    uddot_ilc, angaccel_ilc = u_ilc

    snap = -self.k1.dot(pos - pos_des) - self.k2.dot(vel - vel_des) - self.k3.dot(acc - acc_des) -self.k4.dot(jerk - jerk_des) + snap_des + u_ilc

    uddot = snap.dot(z) + self.int_u * zdot.T.dot(zdot)
    u_ang_accel = (1.0 / self.int_u) * np.cross(z, snap - 2 * self.int_udot * zdot)

    if integrate:
      self.int_u += self.int_udot * dt
      self.int_udot += uddot * dt

    return np.hstack((self.int_u, u_ang_accel))
