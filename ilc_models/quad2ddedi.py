import numpy as np

from ilc_models.base import g, g2
from ilc_models.quad2d import Quad2D

K1 = 840
K2 = 480
K3 = 120
K4 = 16

class Quad2DDEDI(Quad2D):
  """
    state is (pos, vel, theta, omega, u, udot)
    control is (uddot, angular acceleration)
  """
  n_state = 8
  n_control = 2
  n_out = 2

  state_labels = [
    "Position X",
    "Position Z",
    "Velocity X",
    "Velocity Z",
    "Roll",
    "Roll Velocity",
    "Thrust",
    "Thrust Velocity"
  ]

  control_labels = [
    "Thrust Accel",
    "Roll Accel"
  ]

  control_normalization = np.array((1e-3, 1e-2))

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reset()

    self.k1 = K1 * np.eye(2)
    self.k2 = K2 * np.eye(2)
    self.k3 = K3 * np.eye(2)
    self.k4 = K4 * np.eye(2)

  def reset(self):
    self.int_udot = 0
    self.int_u = g
    self.zs = []

  def get_ilc_state(self, state, ind):
    return np.hstack((state, self.zs[ind]))

  def get_ABCD(self, state, control, dt):
    X = slice(0, 2)
    V = slice(2, 4)
    TH = slice(4, 5)
    OM = slice(5, 6)
    U = slice(6, 7)
    UDOT = slice(7, 8)
    UDDOT = slice(0, 1)
    AA = slice(1, 2)
    SNAP_FF = slice(0, 2)

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    pos = state[X]
    vel = state[V]
    theta = state[TH][0]
    om = state[OM][0]
    u = state[U][0]
    udot = state[UDOT][0]

    z = np.array((-np.sin(theta), np.cos(theta)))
    dz_dth = np.array((-np.cos(theta), -np.sin(theta)))
    zdot = dz_dth * om

    acc = u * z - g2
    jerk = udot * z + u * zdot
    snap = -self.k1.dot(pos - self.pos_des) - self.k2.dot(vel - self.vel_des) - self.k3.dot(acc - self.acc_des) -self.k4.dot(jerk - self.jerk_des) + self.snap_des

    dzdot_dth = np.array((np.sin(theta) * om, -np.cos(theta) * om))
    dzdot_dom = dz_dth

    da_dth = u * dz_dth
    da_du = z

    dj_dth = udot * dz_dth + u * dzdot_dth
    dj_dom = u * dzdot_dom
    dj_du = zdot
    dj_dudot = z

    ds_dx =  -self.k1
    ds_dv =  -self.k2
    ds_dth = -self.k3.dot(da_dth) - self.k4.dot(dj_dth)
    ds_dom = -self.k4.dot(dj_dom)
    ds_du =    -self.k3.dot(da_du) - self.k4.dot(dj_du)
    ds_dudot = -self.k4.dot(dj_dudot)

    duddot_dx = ds_dx.T.dot(z)
    duddot_dv = ds_dv.T.dot(z)
    duddot_dth = ds_dth.T.dot(z) + snap.T.dot(dz_dth)
    duddot_dom = ds_dom.T.dot(z) + 2 * u * zdot.T.dot(dzdot_dom)
    duddot_du = ds_du.T.dot(z) + zdot.T.dot(zdot)
    duddot_dudot = ds_dudot.T.dot(z)

    u_fact = 1.0 / u
    dalpha_dx = u_fact * np.cross(z, ds_dx)
    dalpha_dv = u_fact * np.cross(z, ds_dv)
    dalpha_dth = u_fact * (np.cross(dz_dth, snap) + np.cross(z, ds_dth) - 2 * udot * np.cross(dz_dth, zdot))
    dalpha_dom = u_fact * (np.cross(z, ds_dom) - 2 * udot * np.cross(z, dzdot_dom))
    dalpha_du = u_fact * np.cross(z, ds_du) - (1.0 / u ** 2) * (np.cross(z, snap) - 2 * udot * np.cross(z, zdot))
    dalpha_dudot = u_fact * (np.cross(z, ds_dudot) - 2 * np.cross(z, zdot))

    A[X, X] = np.eye(2)
    A[X, V] = dt * np.eye(2)
    A[V, V] = np.eye(2)
    A[V, TH] = np.expand_dims(dt * da_dth, 1)
    A[V, U] = np.expand_dims(dt * da_du, 1)
    A[TH, TH] = 1
    A[TH, OM] = dt
    A[OM, OM] = 1
    A[U, U] = 1
    A[U, UDOT] = dt
    A[UDOT, UDOT] = 1

    B[OM, AA] = dt
    B[UDOT, UDDOT] = dt

    K_x = np.zeros((self.n_control, self.n_state))
    K_u = np.zeros((self.n_control, self.n_control))

    K_x[UDDOT, X] = duddot_dx
    K_x[UDDOT, V] = duddot_dv
    K_x[UDDOT, TH] = duddot_dth
    K_x[UDDOT, OM] = duddot_dom
    K_x[UDDOT, U] = duddot_du
    K_x[UDDOT, UDOT] = duddot_dudot

    K_x[AA, X] = dalpha_dx
    K_x[AA, V] = dalpha_dv
    K_x[AA, TH] = dalpha_dth
    K_x[AA, OM] = dalpha_dom
    K_x[AA, U] = dalpha_du
    K_x[AA, UDOT] = dalpha_dudot

    duddot_ds = z
    dalpha_ds = u_fact * np.array((-z[1], z[0]))

    K_u[UDDOT, UDDOT] = 1
    K_u[AA, AA] = 1

    A = A + B.dot(K_x)
    B = B.dot(K_u)

    C[X, X] = np.eye(2)

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

    snap = -self.k1.dot(pos - pos_des) - self.k2.dot(vel - vel_des) - self.k3.dot(acc - acc_des) -self.k4.dot(jerk - jerk_des) + snap_des

    uddot = snap.dot(z) + self.int_u * zdot.T.dot(zdot) + uddot_ilc
    u_ang_accel = (1.0 / self.int_u) * np.cross(z, snap - 2 * self.int_udot * zdot) + angaccel_ilc

    if integrate:
      self.int_u += self.int_udot * dt
      self.int_udot += uddot * dt

    return np.hstack((self.int_u, u_ang_accel))
