import numpy as np

from ilc_models.base import g, g2
from ilc_models.quad2d import Quad2D

class Quad2DDEDI(Quad2D):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.int_udot = 0
    self.int_u = 0

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
