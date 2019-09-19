import numpy as np

from ilc_models.quadlin import QuadLin

class QuadLinPos(QuadLin):
  """ The output we want to track here is position instead of acceleration. """

  control_normalization = 1e-4

  def get_ABCD(self, state, control, dt):
    if not self.use_feedback:
      self.K_pos *= 0
      self.K_att *= 0

    A = np.eye(self.n_state)
    np.fill_diagonal(A[:, 1:], dt * np.ones(self.n_state - 1))
    A[self.n_state - 1, 0] =    -dt * self.K_att[0] * self.K_pos[0]
    A[self.n_state - 1, 1] =    -dt * self.K_att[0] * self.K_pos[1]
    A[self.n_state - 1, 2] =    -dt * self.K_att[0]
    A[self.n_state - 1, 3] = 1 - dt * self.K_att[1]

    B = np.array(( (0,), (0,), (0,), (dt,) ))
    C = np.array(( (1, 0, 0, 0), ))
    D = np.zeros((1, 1))

    return A, B, C, D
