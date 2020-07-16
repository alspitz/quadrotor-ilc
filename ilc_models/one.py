import numpy as np

from ilc_models.base import ILCBase

class One(ILCBase):
  """
    state is (pos, vel)
    control is (acc)
  """
  n_state = 2
  n_control = n_control_sys = 1
  n_out = 1

  k_pos = 40
  k_vel = 20

  control_labels = sys_control_labels = ["Acc"]

  def get_feedback_response(self, state, control, dt):
    K_x = np.zeros((self.n_control_sys, self.n_state))
    K_u = np.zeros((self.n_control_sys, self.n_control))

    # d output / d pos, i.e. how the output changes w.r.t. the position.
    K_x[0, 0] = -self.k_pos
    # d output / d vel, i.e. how the output changes w.r.t. the velocity.
    K_x[0, 1] = -self.k_vel

    # ILC output gets directly added to feedback output.
    K_u[0, 0] = 1

    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    A = np.array(( (1, dt), (0, 1,), ))
    B = np.array(( (0,), (dt,), ))
    C = np.array(( (1, 0,), ))
    D = np.array(( (0,), ))

    if self.use_feedback:
      K_x, K_u = self.get_feedback_response(state, control, dt)
      A += B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos, vel = x = np.zeros(2)
    xs = [x.copy()]
    for i in range(int(round(t_end / dt))):
      acc = fun(x)
      pos += vel * dt
      vel += acc * dt
      x = np.hstack((pos, vel))
      xs.append(x)
    return np.array(xs)

  def feedback(self, x, pos_des, vel_des, u_ilc, **kwargs):
    K_pos = np.array((self.k_pos, self.k_vel))
    pos_vel_des = np.hstack((pos_des, vel_des))
    return -K_pos.dot(x - pos_vel_des) + u_ilc
