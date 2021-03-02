import numpy as np

from python_utils import mathu

from scipy.spatial.transform import Rotation

from ilc_models.base import g, g3
from ilc_models.quad3dfl import Quad3DFL

K1 = 1040
K2 = 600
K3 = 190
K4 = 25

class Quad3DFLV(Quad3DFL):
  """ This system is augmented with the two virtual control integrators
      used in dynamic extension to delay the appearance of the thrust
      control input u. """
  n_state = 14
  n_control = 4
  n_out = 3

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
    "Thrust",
    "Thrust Velocity"
  ]

  control_labels = [
    "Thrust Accel",
    "Roll Accel",
    "Pitch Accel",
    "Yaw Accel"
  ]


  control_normalization = np.array((1e-3, 1e-2, 1e-2, 1e-2))

  def get_feedback_response(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    RPY = slice(6, 9)
    OM = slice(9, 12)
    Z1 = slice(12, 13)
    Z2 = slice(13, 14)
    V1 = slice(0, 1)
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

    skew_angvel_w = mathu.skew_matrix(rot.apply(angvel))
    z_dot = skew_angvel_w.dot(z)

    #u, udot = self.int_u, self.int_udot
    u, udot = state[Z1][0], state[Z2][0]

    k1 = K1 * np.eye(3) / self.duration ** 4
    k2 = K2 * np.eye(3) / self.duration ** 3
    k3 = K3 * np.eye(3) / self.duration ** 2
    k4 = K4 * np.eye(3) / self.duration ** 1

    dsdu = -k3.dot(z) - k4.dot(z_dot)
    dv1du = dsdu.T.dot(z) + z_dot.T.dot(z_dot)

    dsdudot = -k4.dot(z)
    dv1dudot = dsdudot.T.dot(z)

    K_x = np.zeros((self.n_control_sys, self.n_state))

    K_x[V1, X] = -k1.T.dot(z)
    K_x[V1, V] = -k2.T.dot(z)
    K_x[V1, Z1] = dv1du
    K_x[V1, Z2] = dv1dudot

    self.int_u = u
    self.int_udot = udot
    fb_resp, K_u_old = Quad3DFL.get_feedback_response(self, state, control, dt, nou=True)

    K_x[AA, X] = fb_resp[AA, X]
    K_x[AA, V] = fb_resp[AA, V]
    K_x[AA, RPY] = fb_resp[AA, RPY]
    K_x[AA, OM] = fb_resp[AA, OM]

    skew_z = mathu.skew_matrix(z)

    dzdotdang = -skew_z.dot(rot.as_matrix())

    start_acc = u * z - g3
    start_jerk = u * z_dot + udot * z

    pos_err = pos - self.pos_des
    vel_err = vel - self.vel_des
    acc_err = start_acc - self.acc_des
    jerk_err = start_jerk - self.jerk_des

    dzdotdrpy = skew_angvel_w.dot(dzdrpy)

    djerkdrpy = u * dzdotdrpy + udot * dzdrpy
    daccdrpy = u * dzdrpy

    dsnapdrpy = -k3.dot(daccdrpy) - k4.dot(djerkdrpy)

    snap = -k1.dot(pos_err) - k2.dot(vel_err) - k3.dot(acc_err) - k4.dot(jerk_err) + self.snap_des
    dv1drpy = dsnapdrpy.T.dot(z) + snap.dot(dzdrpy) + 2 * u * z_dot.T.dot(dzdotdrpy)

    #print(snap, z)
    #print(dsnapdrpy[:, 0])
    ##print(daccdrpy[:, 0])
    ##print(dzdrpy[:, 0])
    #input()

    djerkdang = u * dzdotdang
    dsnapdang = -k4.dot(djerkdang)
    dv1dang = dsnapdang.T.dot(z) + 2 * u * z_dot.T.dot(dzdotdang)

    K_x[V1, RPY] = dv1drpy
    K_x[V1, OM] = dv1dang

    dzddotdu = (1.0 / u) * (dsdu - dv1du * z) - (1.0 / u ** 2) * (snap - 2 * udot * z_dot)
    dalphadu = rot.inv().apply(skew_z.dot(dzddotdu))

    dzddotdudot = (1.0 / u) * (dsdudot - 2 * z_dot)
    dalphadudot = rot.inv().apply(skew_z.dot(dzddotdudot))

    K_x[AA, Z1] = np.array([dalphadu]).T
    K_x[AA, Z2] = np.array([dalphadudot]).T

    # TODO XXX Should the below be d u / d uddot ??????????? i.e. 0?! Try this
    K_u = np.zeros((self.n_control_sys, self.n_control))
    #K_u[0, 2] = 1
    #K_u[1, 1] = -1.0 / u
    #K_u[2, 0] = 1.0 / u
    #K_u[AA, AA] = np.eye(3)

    return K_x, K_u

  def get_ABCD(self, state, control, dt):
    X = slice(0, 3)
    V = slice(3, 6)
    RPY = slice(6, 9)
    OM = slice(9, 12)
    Z1 = slice(12, 13)
    Z2 = slice(13, 14)
    V1 = slice(0, 1)
    AA = slice(1, 4)

    pos = state[X]
    vel = state[V]

    u = state[Z1][0]
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

    A[V, Z1] = dt * np.array([z]).T
    A[Z1, Z1] = A[Z2, Z2] = 1
    A[Z1, Z2] = dt

    B[OM, AA] = dt * np.eye(3)
    B[Z2, V1] = dt

    # Trying 2D mats.......
    #ct = np.cos(rpy[0])
    #st = np.sin(rpy[0])
    #A[V, 6:7] = u * dt * np.array(((0, -ct, -st),)).T
    #A[6, 9] = dt
    #########################33

    C[X, X] = np.eye(3)

    if self.use_feedback:
      ind = self.iter - 1 if self.iter >= len(self.zs) - 1 else self.iter
      u, udot = self.zs[ind]

      oldu = self.int_u
      oldudot = self.int_udot

      self.int_u = u
      self.int_udot = udot

      K_x, K_u = self.get_feedback_response(state, control, dt)

      self.int_u = oldu
      self.int_udot = oldudot

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

    y_b = np.cross(z_b, np.array((1, 0, 0)))
    y_b /= np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)

    ang_vel_body = np.array((ang_vel.dot(x_b), ang_vel.dot(y_b), ang_vel.dot(z_b)))

    state = np.hstack((pos, vel, z_b, ang_vel_body, g, 0))
    control = np.hstack((0, np.zeros(3)))

    return state, control

  def feedback(self, x, dt, pos_des, vel_des, acc_des, jerk_des, snap_des, u_ilc, integrate=True, **kwargs):
    pos = x[:3]
    vel = x[3:6]
    rpy = x[6:9]
    ang_vel = x[9:]

    self.zs.append(np.array((self.int_u, self.int_udot)))

    rot = Rotation.from_euler('ZYX', rpy[::-1])
    ang_vel = x[9:]
    rot_m = rot.as_matrix()
    self.x_b_act = rot_m[:, 0]
    self.y_b_act = rot_m[:, 1]
    self.z_b_act = z_b_act = rot_m[:, 2]

    ang_world = rot.apply(ang_vel)
    z_b_dot_act = np.cross(ang_world, z_b_act)

    u = self.int_u
    udot = self.int_udot

    if self.model_drag:
      drag_dist_control = self.drag_dist
    else:
      drag_dist_control = 0

    start_acc = u * z_b_act - g3 - drag_dist_control * vel
    start_jerk = u * z_b_dot_act + udot * z_b_act - drag_dist_control * start_acc

    self.z = z_b_act
    self.acc = start_acc

    pos_err = pos - pos_des
    vel_err = vel - vel_des
    acc_err = start_acc - acc_des
    jerk_err = start_jerk - jerk_des

    k1 = K1 * np.eye(3) / self.duration ** 4
    k2 = K2 * np.eye(3) / self.duration ** 3
    k3 = K3 * np.eye(3) / self.duration ** 2
    k4 = K4 * np.eye(3) / self.duration ** 1

    # Linear controller
    snap = -k1.dot(pos_err) - k2.dot(vel_err) - k3.dot(acc_err) - k4.dot(jerk_err) + snap_des

    self.snap = snap

    v1 = snap.dot(z_b_act) + u * z_b_dot_act.dot(z_b_dot_act) + drag_dist_control * start_jerk.dot(z_b_act) + u_ilc[0]

    self.v1 = v1

    # approx. thrust delay
    #self.int_udot = 0.065 * (snap.dot(z_b_act) + u * z_b_dot_act.dot(z_b_dot_act))

    udot = self.int_udot

    # Here we don't include v1 * z_b_act, because when crossed with Z, this term is zero.
    z_ddot = (1 / u) * (snap - 2 * udot * z_b_dot_act + drag_dist_control * start_jerk)

    ang_acc_world = np.cross(z_b_act, z_ddot) - (ang_world.dot(z_b_act)) * np.cross(z_b_act, ang_world)
    ang_acc_body = rot.inv().apply(ang_acc_world) + u_ilc[1:]

    u_ret = self.int_u

    if integrate:
      self.int_u += self.int_udot * dt
      self.int_udot += v1 * dt

    control = np.hstack((u_ret, ang_acc_body))
    return control

  def get_ilc_state(self, state, ind):
    return np.hstack((state, self.zs[ind]))
