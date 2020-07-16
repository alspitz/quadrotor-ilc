import numpy as np

from ilc_models.quadlin import QuadLin

class QuadLinPos(QuadLin):
  """ The output we want to track here is position instead of acceleration. """

  control_normalization = 1e-4

  def get_feedback_response(self, state, control, dt):
    K_x = np.zeros((self.n_control_sys, self.n_state))
    K_u = np.zeros((self.n_control_sys, self.n_control))

    K_x[0, 0] = -self.K_pos[0] * self.K_att[0]
    K_x[0, 1] = -self.K_pos[1] * self.K_att[0]
    K_x[0, 2] = -self.K_att[0]
    K_x[0, 3] = -self.K_att[1]

    # ILC output gets directly added to feedback output.
    K_u[0, 0] = 1

    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    A = np.eye(self.n_state)
    np.fill_diagonal(A[:, 1:], dt * np.ones(self.n_state - 1))
    B = np.array(( (0,), (0,), (0,), (dt,) ))
    C = np.array(( (1, 0, 0, 0), ))
    D = np.zeros((1, 1))

    if self.use_feedback:
      K_x, K_u = self.get_feedback_response(state, control, dt)
      A += B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D
