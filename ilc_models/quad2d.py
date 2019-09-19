import numpy as np

from ilc_models.base import ILCBase, g, g2

class Quad2D(ILCBase):
  """
    state is (pos, vel, theta, omega)
    control is (u, angular acceleration)
  """
  n_state = 6
  n_control = 2
  n_out = 2

  control_normalization = np.array((1e-1, 1e-3))

  K_pos = np.array((
    (8, 0, 16, 0),
    (0, 8, 0, 4)
  )) / 4
  K_att = np.array((200, 60))

  use_snap = False

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.drag_dist = kwargs['drag_dist']
    self.thrust_dist = kwargs['thrust_dist']

  def get_ABCD(self, state, control, dt):
    X = slice(0, 2)
    V = slice(2, 4)
    TH = slice(4, 5)
    OM = slice(5, 6)
    U = slice(0, 1)
    AA = slice(1, 2)

    theta = state[TH][0]
    u = control[U][0]

    ct = np.cos(theta)
    st = np.sin(theta)

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(2)
    A[V, V] = np.eye(2)
    A[X, V] = dt * np.eye(2)
    A[V, TH] = u * dt * np.array(((-ct, -st),)).T
    A[TH, TH] = A[OM, OM] = 1
    A[TH, OM] = dt

    B[V, U] = dt * np.array(((-st, ct),)).T
    B[OM, AA] = dt

    C[X, X] = np.eye(2)

    if self.use_feedback:
      K_x = np.zeros((self.n_control, self.n_state))
      K_u = np.zeros((self.n_control, self.n_control))

      pos_vel = state[:4]
      a = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + g2
      adota = a.T.dot(a)
      adir = a / np.sqrt(adota)

      K_x[U, X] = adir.dot(-self.K_pos[:, X])
      K_x[U, V] = adir.dot(-self.K_pos[:, V])

      K_x[AA, X] = self.K_att[0] * (a[1] * self.K_pos[0, X] - a[0] * self.K_pos[1, X]) / adota
      K_x[AA, V] = self.K_att[0] * (a[1] * self.K_pos[0, V] - a[0] * self.K_pos[1, V]) / adota
      K_x[AA, TH] = -self.K_att[0]
      K_x[AA, OM] = -self.K_att[1]

      K_u[U, U] = 1
      K_u[AA, AA] = 1

      A = A + B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, u_ff, angaccel_des):
    pos_vel = x[:4]
    theta = x[4]
    angvel = x[5]

    accel_des = -self.K_pos.dot(pos_vel - np.hstack((pos_des, vel_des))) + acc_des + g2
    a_norm = np.linalg.norm(accel_des)
    z_axis_des = accel_des / a_norm
    theta_des = np.arctan2(z_axis_des[1], z_axis_des[0]) - np.pi / 2

    theta_err = theta - theta_des
    angvel_error = angvel - angvel_des
    u_ang_accel = -self.K_att.dot(np.hstack((theta_err, angvel_error))) + angaccel_des

    return np.hstack((a_norm + u_ff - g, u_ang_accel,))

  def feedforward(self, pos, vel, acc, jerk, snap):
    acc_vec = acc + g2
    u = np.linalg.norm(acc_vec)

    if u < 1e-3:
      print("WARNING: acc norm too low!")

    z_b      = (1.0 / u) * acc_vec
    z_b_dot  = (1.0 / u) * (jerk - z_b.dot(jerk) * z_b)
    z_b_ddot = (1.0 / u) * (snap - (snap.dot(z_b) + jerk.dot(z_b_dot)) * z_b - 2 * jerk.dot(z_b) * z_b_dot)

    theta = np.arctan2(-z_b[0], z_b[1])
    ang_vel = np.cross(z_b, z_b_dot)
    ang_acc = np.cross(z_b, z_b_ddot)

    # TODO HMM XXX
    if self.use_feedback:
      u = g
      if self.use_snap:
        ang_acc *= 0

    state = np.array((pos[0], pos[1], vel[0], vel[1], theta, ang_vel))
    control = np.array((u, ang_acc))

    return state, control

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

      if u < 0:
        u = 0
        print("WARNING: THRUST IS NEGATIVE!")

      st = np.sin(theta)
      ct = np.cos(theta)

      acc = self.thrust_dist * u * np.array((-st, ct)) - g2 - self.drag_dist * vel

      pos += vel * dt
      vel += acc * dt
      theta += angvel * dt
      angvel += angaccel * dt

      x = np.hstack((pos.copy(), vel.copy(), theta, angvel))
      xs.append(x)

    return np.array(xs)
