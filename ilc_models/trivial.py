import numpy as np

from ilc_models.base import ILCBase

class Trivial(ILCBase):
  """
    state is pos
    control is vel

    x_{t+1} = x_t + dt*(u_t - k_vel (x_t - x_td))
            = x_t + dt*(u_t - k_vel*x_t + k_vel*x_td)
            = x_t + dt*u_t - dt*k_vel*x_t + dt*k_vel*x_td
            = (1 - dt*k_vel)*x_t + dt*u_t + dt*k_vel*x_td

    y_t = u_t - k_vel(x_t - x_td)
        = u_t - k_vel*x_t + k_vel*x_td
        = -k_vel*x_t + u_t + k_vel*x_td

    dx_{t+1}/dx = A = 1 - dt*k_vel
    dx_{t+1}/du = B = dt

    dy/dx = C = -k_vel
    dy/du = D = 1

    x_{t+1} = Ax + Bu + dt*k_vel*x_td
    y = Cx + Du + k_vel*x_td

    y_{t+1} = C(Ax + Bu_t + dt*k_vel*x_td) + Du_{t+1} + k_vel*x_td
            = C((1 - dt*k_vel)*x + dt*u_t + dt*k_vel*x_td) + u_{t+1} + k_vel*x_td
            = C(x - dt*k_vel*x + dt*u_t + dt*k_vel*x_td) + u_{t+1} + k_vel*x_td
           ?= C(x + dt*u_t) + u_{t+1} + k_vel*x_td
            = -k_vel*x - k_vel*dt*u_t + u_{t+1} + k_vel*x_td
           ?= -k_vel*dt*u_t + u_{t+1}

    # Not considering affine terms...
    y_{t+1} = C(Ax + Bu_t) + Du_{t+1}
            = C((1 - dt*k_vel)*x + dt*u_t) + u_{t+1}
            = C(x - dt*k_vel*x + dt*u_t) + u_{t+1}
            = -k_vel*x + dt*k_vel*k_vel*x - dt*k_vel*u_t + u_{t+1}
            = -k_vel*dt*u_t + u_{t+1}

    y_{t+1} = CAx + CBu_t + Du_{t+1}
  """
  n_state = 1
  n_control = n_control_sys = 1
  n_out = 1

  control_labels = sys_control_labels = ["Vel"]

  k_pos = 100

  def get_feedback_response(self, state, control, dt):
    K_x = np.zeros((self.n_control_sys, self.n_state))
    K_u = np.zeros((self.n_control_sys, self.n_control))

    K_x[0, 0] = -self.k_pos
    K_u[0, 0] = 1

    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    A = np.array(( (1.0,), ))
    B = np.array(( (dt,), ))
    C = np.array(( (1.0,), ))
    D = np.array(( (0.0,), ))

    if self.use_feedback:
      K_x, K_u = self.get_feedback_response(state, control, dt)
      A += B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def simulate(self, t_end, fun, dt):
    vel = x = np.zeros(1)
    xs = [x.copy()]
    for i in range(int(round(t_end / dt))):
      acc = fun(x)
      vel += acc * dt
      x = vel.copy()
      xs.append(x)
    return np.array(xs)

  def feedback(self, x, pos_des, u_ilc, **kwargs):
    return np.array( -self.k_pos * (x - 0) + u_ilc,)
