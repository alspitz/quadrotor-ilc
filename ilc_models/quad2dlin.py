import numpy as np

from ilc_models.base import ILCBase, g2

class Quad2DLin(ILCBase):
  """
    Same as Quad2D but no sines and cosines
  """
  n_state = 6
  n_control = 2
  n_out = 2

  control_normalization = np.array((1e-1, 1e-3))

  g_vec = g2

  def get_ABCD(self, state, control, dt):
    X = slice(0, 2)
    V = slice(2, 4)
    TH = slice(4, 5)
    OM = slice(5, 6)
    U = slice(0, 1)
    AA = slice(1, 2)

    theta = state[TH][0]
    u = control[U][0]

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(2)
    A[V, V] = np.eye(2)
    A[X, V] = dt * np.eye(2)
    A[V, TH] = u * dt * np.array(((-1, 0),)).T
    A[TH, TH] = A[OM, OM] = 1
    A[TH, OM] = dt

    B[V, U] = dt * np.array(((-theta, 1),)).T
    B[OM, AA] = dt

    C[X, X] = np.eye(2)

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos = np.zeros(2)
    vel = np.zeros(2)
    theta = 0
    angvel = 0

    x = np.zeros(6)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      u, angaccel = u_out

      acc = u * np.array((-theta, 1)) - g2

      pos += vel * dt
      vel += acc * dt
      theta += angvel * dt
      angvel += angaccel * dt

      x = np.hstack((pos.copy(), vel.copy(), theta, angvel))
      xs.append(x)

    return np.array(xs)
