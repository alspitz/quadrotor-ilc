import numpy as np

g = 9.81
g2 = np.array((0, g))
g3 = np.array((0, 0, g))

class ILCBase(object):
  control_normalization = 1
  constant_ilc_mats = False
  saved_ilc = None

  def __init__(self, **kwargs):
    self.use_feedback = kwargs['feedback']
    self.reset()

  def reset(self):
    pass

  def get_ilc_state(self, state, ind):
    return state

  def get_learning_operator(self, dt, states, controls, desired_pos, desired_vel, desired_acc, desired_jerk, desired_snap):
    assert len(desired_pos) == len(desired_vel) == len(desired_acc) == len(desired_jerk) == len(desired_snap) == len(controls) == len(states)

    if self.constant_ilc_mats and self.saved_ilc is not None:
      return self.saved_ilc

    N = len(states) - 1

    As = []
    Bs = []
    Cs = []
    Ds = []

    K_xs = []
    K_us = []

    # First we linearize the dynamics around the controls and resulting states.
    for i in range(N + 1):
      state = states[i]

      if i < N:
        control_ind = i
      else:
        control_ind = i - 1

      if not self.constant_ilc_mats:
        state = self.get_ilc_state(state, control_ind)

      control = controls[i]

      self.pos_des = desired_pos[i]
      self.vel_des = desired_vel[i]
      self.acc_des = desired_acc[i]
      self.jerk_des = desired_jerk[i]
      self.snap_des = desired_snap[i]

      self.iter = i
      A, B, C, D = self.get_ABCD(state, control, dt)

      As.append(A)
      Bs.append(B)
      Cs.append(C)
      Ds.append(D)

      K_x, K_u = self.get_feedback_response(state, control, dt)

      K_xs.append(K_x)
      K_us.append(K_u)

      # TODO: Use D
      assert np.all(D == 0)

    calCBpD = np.zeros((N * self.n_out, N * self.n_control))
    G = np.zeros((N * self.n_control, N * self.n_control))

    Apowers = [np.eye(self.n_state) for _ in range(N)]
    last_Apowers = np.array(Apowers).copy()
    last2_Apowers = np.array(Apowers).copy()
    for i in range(N):
      for j in range(N - i):
        row_ind = i + j
        col_ind = j

        calCBpD[self.n_out *     row_ind : self.n_out * (row_ind + 1),
                self.n_control * col_ind : self.n_control * (col_ind + 1)] = Cs[row_ind + 1].dot(Apowers[j].dot(Bs[col_ind]))

        if row_ind >= col_ind + 1:
          G[self.n_control * row_ind : self.n_control * (row_ind + 1),
            self.n_control * col_ind : self.n_control * (col_ind + 1)] = K_xs[row_ind].dot(last_Apowers[j].dot(Bs[col_ind + 1]))
        elif row_ind == col_ind:
          G[self.n_control * row_ind : self.n_control * (row_ind + 1),
            self.n_control * col_ind : self.n_control * (col_ind + 1)] = K_us[row_ind]

        last2_Apowers = last_Apowers.copy()
        last_Apowers[j] = Apowers[j]
        Apowers[j] = As[row_ind + 1].dot(Apowers[j])

      #calCBpD[self.n_out * i : self.n_out * (i + 1), self.n_control * i : self.n_control * (i + 1)] += D

    if self.constant_ilc_mats:
      self.saved_ilc = calCBpD, G

    return calCBpD, G
