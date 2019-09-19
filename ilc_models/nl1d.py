import numpy as np

from ilc_models.base import ILCBase

class NL1D(ILCBase):
  """
    state is (pos, vel, theta, omega)
    control is (angular acceleration)
  """
  n_state = 4
  n_control = 1
  n_out = 1

  c = 100

  def get_ABCD(self, state, control, dt):
    A = np.eye(self.n_state)
    np.fill_diagonal(A[:, 1:], dt * np.ones(self.n_state - 1))
    A[1, 2] = np.cos(state[2] / self.c) * dt
    A[self.n_state - 1, 3] = 1

    B = np.array(( (0,), (0,), (0,), (dt,) ))
    C = np.array(( (1, 0, 0, 0), ))
    D = np.zeros((1, 1))

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos, vel, theta, jerk = x = np.zeros(4)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      snap = u_out[0]

      acc = self.c * np.sin(theta / self.c)

      pos += vel * dt
      vel += acc * dt
      theta += jerk * dt
      jerk += snap * dt

      x = np.hstack((pos.copy(), vel.copy(), theta.copy(), jerk.copy()))
      xs.append(x)

    return np.array(xs)
