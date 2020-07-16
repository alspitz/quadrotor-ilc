import numpy as np

from ilc_models.base import ILCBase

class QuadLin(ILCBase):
  """
    state is (pos, vel, theta, omega)
    control is (angular acceleration)

    om_{t+1} = om_{t} + dt*{-K_att0 * (theta - theta_des) - K_att1 * (om - om_des)}
  """
  n_state = 4
  n_control = n_control_sys = 1
  n_out = 1

  control_labels = sys_control_labels = ["Snap"]

  K_pos = np.array((6, 3))
  K_att = np.array((120, 16))

  control_normalization = 1e-1

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
    C = np.array(( (0, 0, 1, 0), ))
    D = np.zeros((1, 1))

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    pos, vel, acc, jerk = x = np.zeros(4)

    xs = [x.copy()]

    for i in range(int(round(t_end / dt))):
      u_out = fun(x)

      snap = u_out[0]

      pos += vel * dt
      vel += acc * dt
      acc += jerk * dt
      jerk += snap * dt

      x = np.hstack((pos.copy(), vel.copy(), acc.copy(), jerk.copy()))
      xs.append(x)

    return np.array(xs)

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, angaccel_des, u_ilc, **kwargs):
    pos_vel = x[:2]
    theta = x[2]
    angvel = x[3]

    accel_des = -self.K_pos.dot(pos_vel - np.hstack((pos_des, vel_des))) + acc_des
    theta_des = accel_des

    theta_err = theta - theta_des
    angvel_error = angvel - angvel_des
    u_ang_accel = -self.K_att.dot(np.hstack((theta_err, angvel_error))) + angaccel_des + u_ilc

    return np.array((u_ang_accel,))

  def feedforward(self, pos, vel, acc, jerk, snap):
    return np.hstack((pos, vel, acc, jerk)), np.hstack((snap))
