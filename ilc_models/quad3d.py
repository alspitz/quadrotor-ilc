import numpy as np

from python_utils import mathu

from scipy.spatial.transform import Rotation

from ilc_models.base import ILCBase, g, g3
from lqr_gain_match.match_full_state import accel_to_euler_rpy
from python_utils.rigid_body_lie import RigidBody3D

class U_AccelNorm:
  @staticmethod
  def u(a, z):
    return np.linalg.norm(a)

  @staticmethod
  def duda(a, z):
    return a / U_AccelNorm.u(a, z)

  @staticmethod
  def dudz(a, z):
    return np.zeros(3)

class U_AccelProj:
  @staticmethod
  def u(a, z):
    return a.dot(z)

  @staticmethod
  def duda(a, z):
    return z.copy()

  @staticmethod
  def dudz(a, z):
    return a.copy()

class U_AccelZPri:
  @staticmethod
  def u(a, z):
    return a[2] / z[2]

  @staticmethod
  def duda(a, z):
    return np.array((0, 0, 1 / z[2]))

  @staticmethod
  def dudz(a, z):
    return np.array((0, 0, -a[2] / (z[2] ** 2)))

class Delay:
  """
     v dot = - tau * (v - v_des)
  """
  def __init__(self, tau, v):
    self.tau = tau
    self.v = v

  def step(self, dt, v_des):
    self.v += -self.tau * (self.v - v_des) * dt

class Quad3D(ILCBase):
  """
    state is (pos, vel, rpy, angular velocity)
    control is (u, angular acceleration)

    #z_next = z + dt * [angvel]_x[:, 3]
    #angvel_next = omega + dt * angular acceleration

    #obs = accel = u * z axis + g
  """
  n_state = 12
  n_control = 4
  n_out = 3

  n_control_sys = 4

  state_labels = [
    "Position X",
    "Position Y",
    "Position Z",
    "Velocity X",
    "Velocity Y",
    "Velocity Z",
    "Roll",
    "Pitch",
    "Yaw",
    "Roll Velocity",
    "Pitch Velocity",
    "Yaw Velocity",
  ]

  control_labels = [
    "Thrust",
    "Roll Accel",
    "Pitch Accel",
    "Yaw Accel"
  ]

  sys_control_labels = [
    "Thrust",
    "Roll Accel",
    "Pitch Accel",
    "Yaw Accel"
  ]

  g_vec = g3

  K_pos = np.array((
    #(8.0, 0, 0, 6.0, 0, 0),
    #(0, 8.0, 0, 0, 6.0, 0),
    (5.47368, 0, 0, 3.15789, 0, 0),
    (0, 5.47368, 0, 0, 3.15789, 0),
    (0, 0, 5.47368, 0, 0, 3.15789)
  ))

  #K_att = np.array((
  #  (120, 0, 0, 16, 0, 0),
  #  (0, 120, 0, 0, 16, 0),
  #  (0, 0, 60, 0, 0, 12)
  #))
  K_att = np.array((
    (190, 0, 0, 25, 0, 0),
    (0, 190, 0, 0, 25, 0),
    (0, 0, 30, 0, 0, 3)
  ))

  control_normalization = np.array((0.1, 1e-2, 1e-2, 1e-2))

  def __init__(self, **kwargs):
    ILCBase.__init__(self, **kwargs)

    self.model_drag = kwargs['model_drag']
    self.drag_dist = kwargs['drag_dist']
    self.thrust_dist = kwargs['thrust_dist']
    self.angaccel_dist = kwargs['angaccel_dist']
    self.periodic_accel_dist_mag = kwargs['periodic_accel_dist_mag']
    self.periodic_accel_dist_periods = kwargs['periodic_accel_dist_periods']

    self.delay_control = kwargs['delay_control']
    self.delay_timeconstant = kwargs['delay_timeconstant']
    self.positive_thrust_only = kwargs['positive_thrust_only']
    self.accel_limit = kwargs['accel_limit']
    self.angaccel_limit = kwargs['angaccel_limit']

    ttype = kwargs['cascaded_thrust']
    if ttype == 'norm':
      self.U = U_AccelNorm
    elif ttype == 'project':
      self.U = U_AccelProj
    elif ttype == "maintain-z":
      self.U = U_AccelZPri
    else:
      assert False

    self.mixer = np.array((
      (0.25, 0.25, 0.25, 0.25),
      (-0.25, 0.25, 0.25, -0.25),
      (-0.25, 0.25, -0.25, 0.25),
      (-0.25, -0.25, 0.25, 0.25)
    ))

    # To ensure rotor forces are always positive!
    self.mixer[0, :] /= 40

    self.mixer_true = self.mixer.copy()

    if kwargs['limit_motor']:
      self.mixer_true[:, kwargs['limit_motor_ind']] *= kwargs['limit_motor_scale']

    self.mixer_inv = np.linalg.inv(self.mixer)

  def get_feedback_response(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    RPY = slice(6, 9)
    OM = slice(9, 12)
    U = slice(0, 1)
    AA = slice(1, 4)

    pos = state[X]
    vel = state[V]

    rpy = state[RPY]
    angvel = state[OM]
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))

    roll, pitch, yaw = rpy

    dzdrpy = np.array((
      (np.sin(yaw) * np.cos(roll) - np.sin(roll) * np.cos(yaw) * np.sin(pitch), np.cos(roll) * np.cos(yaw) * np.cos(pitch), np.sin(roll) * np.cos(yaw) - np.cos(roll) * np.sin(yaw) * np.sin(pitch)),
      (-np.sin(roll) * np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.cos(roll), np.cos(roll) * np.sin(yaw) * np.cos(pitch), np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll)),
      (-np.cos(pitch) * np.sin(roll), -np.sin(pitch) * np.cos(roll), 0)
    ))

    skew_z = mathu.skew_matrix(z)

    pos_vel = state[:6]
    accel_des = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + g3
    anorm = np.linalg.norm(accel_des)

    K_x = np.zeros((self.n_control, self.n_state))

    dadx = -self.K_pos[:, X]
    dadv = -self.K_pos[:, V]

    duda = self.U.duda(accel_des, z)
    dudz = self.U.dudz(accel_des, z)

    dudx = duda.dot(dadx)
    dudv = duda.dot(dadv)

    K_x[U, X] = dudx
    K_x[U, V] = dudv

    K_x[U, RPY] = dudz.dot(dzdrpy)

    z_des = accel_des / anorm

    dzda = (1.0 / anorm) * (np.eye(3) - np.outer(z_des, z_des))
    dzdx = dzda.dot(dadx)
    dzdv = dzda.dot(dadv)

    # We assume yaw is zero
    #assert abs(rpy[2]) < 1e-6

    deulerdz = np.zeros((3, 3))
    a1 = 1 / np.sqrt(1 - z_des[1] ** 2)
    deulerdz[0, 1] = -a1
    deulerdz[1, 0] = 1 / np.sqrt(1 - (z_des[0] * a1) ** 2)
    deulerdz[1, 1] = z_des[0] * z_des[1] / (np.sqrt(1 - (z_des[0] * a1) ** 2) * ((1 - z_des[1]**2) ** (3.0/2)))

    #des_phi = np.arcsin(-z_des[1])
    #des_th = np.arctan2(z_des[0], z_des[2])
    #print(dzdrpy)
    #print(deulerdz)
    #print(-1 / np.cos(des_phi))
    #print(1 / (1 + np.tan(des_th)**2))
    #input()

    #deulerdz[0, 1] = -1 / np.cos(des_phi)
    #deulerdz[1, 0] = 1 / (1 + np.tan(des_th) ** 2)
    #deulerdz[1, 1] = deulerdz[1, 0] * (-z_des[0] / (z_des[2] ** 2))

    K_x[AA, X] = self.K_att[:, :3].dot(deulerdz.dot(dzdx))
    K_x[AA, V] = self.K_att[:, :3].dot(deulerdz.dot(dzdv))

    K_x[AA, RPY] = -self.K_att[:, :3]
    K_x[AA, OM] = -self.K_att[:, 3:6]

    K_u = np.zeros((self.n_control, self.n_control))
    K_u[U, U] = 1
    K_u[AA, AA] = np.eye(3)

    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    RPY = slice(6, 9)
    OM = slice(9, 12)
    U = slice(0, 1)
    AA = slice(1, 4)

    rpy = state[RPY]
    angvel = state[OM]
    rot = Rotation.from_euler('ZYX', rpy[::-1])
    z = rot.apply(np.array((0, 0, 1)))

    if not self.use_feedback:
      u = control[U][0]
    else:
      pos_vel = state[:6]
      accel_des = -self.K_pos.dot(pos_vel - np.hstack((self.pos_des, self.vel_des))) + self.acc_des + g3
      u = self.U.u(accel_des, z)

    roll, pitch, yaw = rpy

    dzdrpy = np.array((
      (np.sin(yaw) * np.cos(roll) - np.sin(roll) * np.cos(yaw) * np.sin(pitch), np.cos(roll) * np.cos(yaw) * np.cos(pitch), np.sin(roll) * np.cos(yaw) - np.cos(roll) * np.sin(yaw) * np.sin(pitch)),
      (-np.sin(roll) * np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.cos(roll), np.cos(roll) * np.sin(yaw) * np.cos(pitch), np.cos(roll) * np.cos(yaw) * np.sin(pitch) + np.sin(yaw) * np.sin(roll)),
      (-np.cos(pitch) * np.sin(roll), -np.sin(pitch) * np.cos(roll), 0)
    ))

    A = np.zeros((self.n_state, self.n_state))
    B = np.zeros((self.n_state, self.n_control))
    C = np.zeros((self.n_out, self.n_state))
    D = np.zeros((self.n_out, self.n_control))

    A[X, X] = np.eye(3)
    A[V, V] = np.eye(3)
    A[X, V] = dt * np.eye(3)

    A[RPY, RPY] = np.eye(3)
    A[OM, OM] = np.eye(3)

    A[V, RPY] = u * dt * dzdrpy
    # Below taken from Tal and Karaman 2018 - Accurate Tracking of ...
    A[RPY, OM] = dt * np.array((
      (1, np.sin(roll) * np.tan(pitch), np.cos(roll) * np.tan(pitch)),
      (0, np.cos(roll), -np.sin(roll)),
      (0, np.sin(roll) / np.cos(pitch), np.cos(roll) / np.cos(pitch))))

    B[V, U] = dt * np.array([z]).T
    B[OM, AA] = dt * np.eye(3)

    # Trying 2D mats.......
    #ct = np.cos(rpy[0])
    #st = np.sin(rpy[0])
    #A[V, 6:7] = u * dt * np.array(((0, -ct, -st),)).T
    #A[6, 9] = dt
    #########################33

    C[X, X] = np.eye(3)

    if self.use_feedback:
      K_x, K_u = self.get_feedback_response(state, control, dt)

      A = A + B.dot(K_x)
      B = B.dot(K_u)

    return A, B, C, D

  def feedforward(self, pos, vel, acc, jerk, snap):
    acc_vec = acc + g3
    u = np.linalg.norm(acc_vec)

    if u < 1e-3:
      print("WARNING: acc norm too low!")

    z_b      = (1.0 / u) * acc_vec
    z_b_dot  = (1.0 / u) * (jerk - z_b.dot(jerk) * z_b)

    ang_vel = np.cross(z_b, z_b_dot)

    z_b_ddot = (1.0 / u) * (snap - (snap.dot(z_b) + jerk.dot(z_b_dot)) * z_b - 2 * jerk.dot(z_b) * z_b_dot) - np.cross(ang_vel, z_b_dot)

    y_b = np.cross(z_b, np.array((1, 0, 0)))
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)

    dp = z_b_ddot.dot(-y_b)
    dq = z_b_ddot.dot( x_b)

    ang_acc_body = np.array((dp, dq, 0))

    ang_vel_body = np.array((ang_vel.dot(x_b), ang_vel.dot(y_b), ang_vel.dot(z_b)))

    state = np.hstack((pos, vel, z_b, ang_vel_body))
    control = np.hstack((u, ang_acc_body))

    return state, control

  def simulate(self, t_end, fun, dt):
    from random import gauss
    vec = np.array([gauss(0, 1) for i in range(3)])
    pos = vec * 0.02 / np.linalg.norm(vec)
    vec = np.array([gauss(0, 1) for i in range(3)])
    vel = vec * 0.05 / np.linalg.norm(vec)

    #pos = np.array((0.00, -0.02, 0.00))
    #vel = np.array((0.00, -0.05, 0.00))

    pos *= 0
    vel *= 0

    rot = Rotation.from_euler('ZYX', np.zeros(3))
    rigid_body = RigidBody3D(pos=pos, vel=vel, quat=np.array((1.0, 0, 0, 0)), ang=np.zeros(3))

    x = np.zeros(12)
    xs = [x.copy()]

    delay = Delay(self.delay_timeconstant, np.zeros(4))

    for i in range(int(round(t_end / dt))):
      time = i * dt

      u_out = fun(x)

      if self.delay_control:
        if not i:
          delay.v = u_out

        u_delayed = delay.v.copy()
        delay.step(dt, u_out)

      else:
        u_delayed = u_out.copy()

      rotor_forces = self.mixer_inv.dot(u_delayed)
      u_mixed = self.mixer_true.dot(rotor_forces)

      u_use, aa_use = u_mixed[0], u_mixed[1:]

      if np.any(np.abs(aa_use) > self.angaccel_limit):
        print("WARNING: Ang accel is very high! (%s; limit %f)" % (aa_use, self.angaccel_limit))
        aa_use = np.clip(aa_use, -self.angaccel_limit, self.angaccel_limit)

      if abs(u_use) > self.accel_limit:
        print("WARNING: Accel is very high! (%f; limit %f)" % (u_use, self.accel_limit))
        u_use = np.clip(u_use, -self.accel_limit, self.accel_limit)

      if self.positive_thrust_only and u_use < 0:
        u_use = 0
        print("WARNING: THRUST IS NEGATIVE")

      vel = rigid_body.get_vel()
      acc = self.thrust_dist * u_use * rot.apply(np.array((0, 0, 1))) - g3 - self.drag_dist * vel

      if abs(self.periodic_accel_dist_mag) > 1e-6:
        acc += self.periodic_accel_dist_mag * np.sin(2 * np.pi * self.periodic_accel_dist_periods * time)

      aa_use *= self.angaccel_dist

      aa_use[2] = 0

      ang_accel_world = rot.apply(aa_use)

      rigid_body.step(dt, acc, ang_accel_world)

      if np.any(np.abs(rigid_body.ang) > 10000):
        rigid_body.ang = np.clip(rigid_body.ang, -10000, 10000)
        print("WARNING: Clipping vehicle ang. vel.")

      if np.any(np.abs(rigid_body.vel) > 100):
        rigid_body.vel = np.clip(rigid_body.vel, -100, 100)
        print("WARNING: Clipping vehicle velocity")

      rot = Rotation.from_matrix(rigid_body.get_rot())

      x = np.hstack((rigid_body.get_pos(), rigid_body.get_vel(), rot.as_euler('ZYX')[::-1], rot.inv().apply(rigid_body.get_ang())))

      xs.append(x)

    return np.array(xs)

  def feedback(self, x, pos_des, vel_des, acc_des, angvel_des, angaccel_des, u_ilc, **kwargs):
    pos_vel = x[:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    pos_vel_error = pos_vel - np.hstack((pos_des, vel_des))

    if self.model_drag:
      drag_dist_control = self.drag_dist
    else:
      drag_dist_control = 0

    # Position Control
    accel_des = -self.K_pos.dot(pos_vel_error) + acc_des + drag_dist_control * pos_vel[3:]

    # Reference Conversion
    euler_des = accel_to_euler_rpy(accel_des, g)
    rot = Rotation.from_euler('ZYX', rpy[::-1])

    z_b = rot.apply(np.array((0, 0, 1)))

    accel_des_vec = accel_des + g3
    u_accel = self.U.u(accel_des_vec, z_b)

    # Attitude Control
    euler_error = rpy - euler_des
    angvel_error = ang_vel - angvel_des
    euler_angvel = np.hstack((euler_error, angvel_error))
    u_ang_accel = -self.K_att.dot(euler_angvel) + angaccel_des

    return np.hstack((u_accel + u_ilc[0], u_ang_accel + u_ilc[1:]))
