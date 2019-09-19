import numpy as np

from ilc_models.base import ILCBase

class One(ILCBase):
  """
    state is (pos, vel)
    control is (acc)
  """
  n_state = 2
  n_control = 1
  n_out = 1

  k_pos = 5
  k_vel = 5

  def get_ABCD(self, state, control, dt):
    if not self.use_feedback:
      self.k_pos = 0
      self.k_vel = 0

    A = np.array(( (1, dt), (-self.k_pos * dt, 1 - self.k_vel * dt,), ))
    B = np.array(( (0,), (dt,), ))
    C = np.array(( (-self.k_pos, -self.k_vel,), ))
    D = np.array(( (1,), ))

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

  def feedback(self, x, pos_des, vel_des, acc_des):
    K_pos = np.array((self.k_pos, self.k_vel))
    pos_vel_des = np.hstack((pos_des, vel_des))
    return -K_pos.dot(x - pos_vel_des) + acc_des
