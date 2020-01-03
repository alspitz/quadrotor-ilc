import argparse

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

from ilc_models import base, trivial, one, quadlin, quadlinpos, nl1d, quad2dlin, quad2d, quad2ddedi, quad3d, quad3dfl, quad3dflv
from polynomial_utils import deriv_fitting_matrix

poly_fit_mat = np.linalg.inv(deriv_fitting_matrix(8))

def get_poly(x, v=0, a=0, j=0, end_pos=1.0, duration=1.0):
  poly = poly_fit_mat.dot(np.array((x, v, a, j, end_pos, 0, 0, 0)))

  divider = 1
  for i in range(8):
    poly[i] /= divider
    divider *= duration

  return poly[::-1]

if __name__  == "__main__":
  system_map = {
                # ilc, DIMS
    'trivial': (trivial.Trivial, 1),
    'simple': (one.One, 1),
    'linear': (quadlin.QuadLin, 1),
    'linearpos': (quadlinpos.QuadLinPos, 1),
    'nl1d' : (nl1d.NL1D, 1),
    '2dposlin': (quad2dlin.Quad2DLin, 2),
    '2dpos':     (quad2d.Quad2D, 2),
    '3d':     (quad3d.Quad3D, 3),
    '2ddedi':     (quad2ddedi.Quad2DDEDI, 2),
    '3ddedi':     (quad3dfl.Quad3DFL, 3),
    '3ddediv':     (quad3dflv.Quad3DFLV, 3),
  }

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--system", type=str, default="simple", choices=system_map.keys(), help="Type of system to simulate.")
  parser.add_argument("--trials", type=int, default=4, help="Number of ILC trials to run.")
  parser.add_argument("--alpha", type=float, default=1.0, help="Percentage of update (0 - 1) to use at each iteration. Lower values increase stability.")
  parser.add_argument("--dt", type=float, default=0.02, help="Size of timestep along trajectory.")
  parser.add_argument("--feedback", "--fb", default=False, action='store_true', help="Apply feedback along the trajectory.")
  parser.add_argument("--noise", default=False, action='store_true', help="Add noise to the position errors fed into ILC.")
  parser.add_argument("--noise-stddev", default=1e-3, type=float, help="Stddev of noise added to the position errors")
  parser.add_argument("--filter", default=False, action='store_true', help="Filter the position errors fed into ILC.")
  parser.add_argument("--relin-time", default=True, action='store_true', help="Use a different linearization point at each time step along the trajectory.")
  parser.add_argument("--relin-iter", default=True, action='store_true', help="Use different linearization points for each iteration.")
  parser.add_argument("--no-relin-time", default=False, dest='relin_time', action='store_false')
  parser.add_argument("--no-relin-iter", default=False, dest='relin_iter', action='store_false')
  parser.add_argument("--dist", default=1.5, type=float, help="Distance to travel.")
  parser.add_argument("--rest-time", default=0.0, type=float, help="Rest time to add to end of desired trajectory.")
  parser.add_argument("--feedforward", "--ff", default=False, action='store_true', help="Use initial feedforward trajectory.")
  parser.add_argument("--thrust-dist", default=1.0, type=float, help="Disturbance used to scale the commanded thrust u.")
  parser.add_argument("--drag-dist", default=0.0, type=float, help="Disturbance used to scale velocity subtracted from acceleration as drag.")
  parser.add_argument("--model-drag", default=False, action='store_true', help="Consider the drag disturbance explicitly in the model.")
  parser.add_argument("--poke", default=False, action='store_true', help="Add a \"poke\" disturbance during the 3D trajectory.")
  parser.add_argument("--poke-strength", default=600, type=float, help="Value of constant poke angular acceleration disturbance.")
  parser.add_argument("--poke-time", default=0.5, type=float, help="Time of poke.")
  parser.add_argument("--poke-duration", default=0.03, type=float, help="Duration of poke.")
  parser.add_argument("--plot", default=False, action='store_true', help="Plot the states for each trial.")
  parser.add_argument("--plot-fb-resp", default=False, action='store_true', help="Plot the feedback response along the final trajectory.")
  parser.add_argument("--plot-controls", default=False, action='store_true', help="Plot the final control inputs used.")
  parser.add_argument("--plot-control-corrections", default=False, action='store_true', help="Plot the final ILC control corrections used.")
  parser.add_argument("--plot-updates", default=False, action='store_true', help="Plot the ILC updates after every iteration.")
  parser.add_argument("--w", default=1e-2, type=float, help="Weight of control update norm minimization.")
  parser.add_argument("--save", default=False, action='store_true', help="Write parameters and trajectories to file.")
  args = parser.parse_args()

  try:
    from tabulate import tabulate

    print(tabulate([
      ["System", args.system],
      ["No. of trials", args.trials],
      ["dt", args.dt],
      ["Trajectory Distance", args.dist],
      ["Control Weight", args.w],
      ["Update weight", args.alpha],
      ["Time-varying linearization?", str(args.relin_time)],
      ["Iter-varying linearization?", args.relin_iter],
      ["Initial Feedforward Traj?", args.feedforward],
      ["Feedback?", args.feedback],
      ["Noise?", args.noise],
      ["Filter?", args.filter],
      ['Thrust Disturbance', args.thrust_dist],
      ['Drag Disturbance', args.drag_dist],
      ['Poke Disturbance?', args.poke],
      ['Model Drag', args.model_drag],
      ['Plot FB Resp?', args.plot_fb_resp],
      ['Plot Updates?', args.plot_updates],
    ], tablefmt='fancy_grid', disable_numparse=True))
  except ImportError:
    pass

  assert args.relin_time or not args.relin_iter
  assert not (args.relin_time and (not args.relin_iter) and (not args.feedforward))

  assert args.feedback or (not args.plot_fb_resp)

  ilc_c, DIMS = system_map[args.system]
  ilc = ilc_c(feedback=args.feedback, drag_dist=args.drag_dist, thrust_dist=args.thrust_dist, model_drag=args.model_drag, dt=args.dt)
  AXIS = 1 if DIMS == 3 else 0

  dt = args.dt
  poly_tend = 1.0
  t_end = poly_tend + args.rest_time

  poly_N = int(round(poly_tend / dt))
  poly_ts = np.linspace(0, poly_tend, poly_N + 1)

  N = int(round(t_end / dt))
  ts = np.linspace(0, t_end, N + 1)

  poke_center = args.poke_time / dt
  poke_steps = args.poke_duration / dt

  print("No. of steps is", N)

  def pad_des(des, val=0):
    """ Returns a desired vectory spanning the full length of time
        i.e. adds the rest time in addition to the poly traj time """
    return np.hstack((des, val * np.ones(N - poly_N)))

  pos_poly = get_poly(0, end_pos=args.dist, duration=poly_tend)
  vel_poly = np.polyder(pos_poly)
  acc_poly = np.polyder(vel_poly)
  jerk_poly = np.polyder(acc_poly)
  snap_poly = np.polyder(jerk_poly)

  poss_des = np.polyval(pos_poly, poly_ts)
  poss_des = pad_des(poss_des, args.dist)

  vels_des = np.polyval(vel_poly, poly_ts)
  vels_des = pad_des(vels_des)

  vels_des_vec = np.zeros((N + 1, DIMS))
  vels_des_vec[:, AXIS] = vels_des

  accels_des = pad_des(np.polyval(acc_poly, poly_ts))
  jerks_des = pad_des(np.polyval(jerk_poly, poly_ts))
  snaps_des = pad_des(np.polyval(snap_poly, poly_ts))

  lifted_control = np.zeros(N * ilc.n_control)
  if (DIMS == 2 or DIMS == 3) and not args.feedback:
    lifted_control[::ilc.n_control] = base.g

  cum_updates = np.zeros(lifted_control.shape)

  if args.feedforward:
    assert hasattr(ilc, 'feedforward')

    ff_states = []

    for i in range(N):
      hods = [poss_des[i], vels_des[i], accels_des[i], jerks_des[i], snaps_des[i]]
      hod_vecs = []
      for hod in hods:
        vec = np.zeros(DIMS)
        vec[AXIS] = hod
        hod_vecs.append(vec)

      state, control = ilc.feedforward(*hod_vecs)
      ff_states.append(state)
      if not args.feedback:
        lifted_control[ilc.n_control * i : ilc.n_control * (i + 1)] = control

  initial_lifted_control = lifted_control.copy()

  if args.system in ['2dpos', 'linearpos', '2dposlin', 'nl1d'] or '3d' in args.system:
    for i in range(min(4, N)):
      poss_des[i] = 0.0

  if args.system in ['linear', '2d', 'linearpos']:
    accels_des[1] = 0.0

  poss_des_vec = np.zeros((N + 1, DIMS))
  poss_des_vec[:, AXIS] = poss_des

  accels_des_vec = np.zeros((N + 1, DIMS))
  accels_des_vec[:, AXIS] = accels_des

  jerks_des_vec = np.zeros((N + 1, DIMS))
  jerks_des_vec[:, AXIS] = jerks_des

  snaps_des_vec = np.zeros((N + 1, DIMS))
  snaps_des_vec[:, AXIS] = snaps_des

  class Controller:
    def __init__(self, lifted_control, poss_des, vels_des, accels_des, jerks_des, snaps_des):
      self.poss_des = poss_des
      self.vels_des = vels_des

      self.controls = []
      self.accs_des = []
      self.jerks_des = []
      self.snaps_des = []
      self.angvels_des = []
      self.angaccels_des = []

      for i in range(N):
        self.controls.append(lifted_control[ilc.n_control * i : ilc.n_control * (i + 1)])

        accel = accels_des[i, :]
        jerk = jerks_des[i, :]
        snap = snaps_des[i, :]

        self.accs_des.append(accel)
        if args.feedforward:
          self.jerks_des.append(jerk)
          self.snaps_des.append(snaps_des[i, :])
        else:
          self.jerks_des.append(jerk * 0)
          self.snaps_des.append(snaps_des[i, :] * 0)

        if args.system in ['linear', 'linearpos']:
          self.angvels_des.append(jerk)

        elif '2d' in args.system or '3d' in args.system:
          if args.model_drag:
            drag_dist_control = args.drag_dist
          else:
            drag_dist_control = 0

          acc_vec = accel + ilc.g_vec + drag_dist_control * vels_des[i]
          u = np.linalg.norm(acc_vec)

          if u < 1e-3:
            print("WARNING: acc norm too low!")


          z_b      = (1.0 / u) * acc_vec

          u_dot = z_b.dot(jerk + drag_dist_control * accel)
          z_b_dot  = (1.0 / u) * (jerk - u_dot * z_b + drag_dist_control * accel)

          u_ddot = snap.dot(z_b) + u * z_b_dot.dot(z_b_dot) + drag_dist_control * jerk.dot(z_b)
          z_b_ddot = (1.0 / u) * (snap - u_ddot * z_b - 2 * u_dot * z_b_dot + drag_dist_control * jerk)

          theta = np.arctan2(-z_b[0], z_b[1])
          angvel = np.cross(z_b, z_b_dot)
          angacc = np.cross(z_b, z_b_ddot)

          if args.feedforward:
            self.angvels_des.append(angvel)
            self.angaccels_des.append(angacc)
          else:
            self.angvels_des.append(angvel * 0)
            self.angaccels_des.append(angacc * 0)

      self.index = 0

      self.compute_feedback_response = False
      self.feedback_responses = []
      self.feedback_responses_ana = []
      self.final_controls = []

    def get(self, x):
      ilc_controls = self.controls[self.index]
      if args.feedback:
        if args.system == 'trivial':
          arg_list = [x, self.vels_des[self.index], ilc_controls[0]]
        elif args.system == 'simple':
          arg_list = [x, self.poss_des[self.index], self.vels_des[self.index], ilc_controls[0]]
        elif args.system in ['linear', 'linearpos']:
          arg_list = [x,
            self.poss_des[self.index],
            self.vels_des[self.index],
            self.accs_des[self.index],
            self.angvels_des[self.index],
            ilc_controls[0]
          ]
        elif args.system in ['3ddedi', '3ddediv']:
          arg_list = [x,
            self.poss_des[self.index],
            self.vels_des[self.index],
            self.accs_des[self.index],
            self.jerks_des[self.index],
            self.snaps_des[self.index],
            ilc_controls[0],
            ilc_controls[1:]
          ]
        else:
          arg_list = [x,
            self.poss_des[self.index],
            self.vels_des[self.index],
            self.accs_des[self.index],
            self.angvels_des[self.index],
            self.angaccels_des[self.index],
            ilc_controls[0],
            ilc_controls[1:]
          ]


        if self.compute_feedback_response:
          feedback = ilc.feedback(*arg_list, integrate=False)

          if args.system == '3ddediv':
            feedback[0] = ilc.v1
            init_snap = ilc.snap
            init_acc = ilc.acc
            init_z = ilc.z

          response = np.zeros((ilc.n_control, ilc.n_state))
          try_eps = 1e-6
          for i in range(len(x)):
            x_try = x.copy()
            x_try[i] += try_eps

            arg_list[0] = x_try
            feedback_try = ilc.feedback(*arg_list, integrate=False)

            #if i == 6:
            #  print(ilc.snap, ilc.z)
            #  print((ilc.snap - init_snap) / try_eps)
            #  #print((ilc.acc - init_acc) / try_eps)
            #  #print((ilc.z - init_z) / try_eps)

            if args.system == '3ddediv':
              feedback_try[0] = ilc.v1

            deriv = (feedback_try - feedback) / try_eps

            response[:, i] = deriv

          if args.system == '3ddediv':
            arg_list[0] = x
            orig_u, orig_udot = ilc.int_u, ilc.int_udot
            ilc.int_u = orig_u + try_eps

            feedback_try = ilc.feedback(*arg_list, integrate=False)
            feedback_try[0] = ilc.v1

            deriv = (feedback_try - feedback) / try_eps
            response[:, 12] = deriv

            ilc.int_u = orig_u
            ilc.int_udot = orig_udot + try_eps

            feedback_try = ilc.feedback(*arg_list, integrate=False)
            feedback_try[0] = ilc.v1

            deriv = (feedback_try - feedback) / try_eps
            response[:, 13] = deriv

            ilc.int_u = orig_u
            ilc.int_udot = orig_udot

          self.feedback_responses.append(response)
          pos_des = self.poss_des[self.index]
          vel_des = self.vels_des[self.index]
          acc_des = self.accs_des[self.index]
          jerk_des = self.jerks_des[self.index]
          snap_des = self.snaps_des[self.index]
          if hasattr(ilc, "get_feedback_response"):
            self.feedback_responses_ana.append(ilc.get_feedback_response(x, pos_des, vel_des, acc_des, jerk_des, snap_des))

        arg_list[0] = x
        #x[6] += np.random.normal(scale=0.0005)
        #if self.compute_feedback_response and args.poke:
          #if 100 < self.index < 150:
        #    arg_list[0][6] += 0.1 * np.random.normal()

        feedback = ilc.feedback(*arg_list)

        # Really this should only run on the last iteration.
        # A "poke" like disturbance.
        if self.poke:
          if poke_center - poke_steps / 2 < self.index < poke_center + poke_steps / 2:
            feedback[1] += args.poke_strength

        self.index += 1
        self.final_controls.append(feedback.copy())
        return feedback

      self.index += 1
      self.final_controls.append(ilc_controls.copy())
      return ilc_controls

  trial_poss = []
  trial_vels = []
  trial_accels = []
  trial_omegas = []
  trial_rpys = []

  trial_controls = []
  trial_control_corrections = []

  for iter_no in range(args.trials):
    controller = Controller(lifted_control, poss_des_vec, vels_des_vec, accels_des_vec, jerks_des_vec, snaps_des_vec)
    controller.compute_feedback_response = iter_no == args.trials - 1 and args.feedback and (args.plot_fb_resp or args.save)
    controller.poke = iter_no == args.trials - 1 and args.poke
    ilc.reset()
    data = ilc.simulate(t_end, controller.get, dt=dt)

    if '3d' in args.system:
      poss_vec = data[:, :3]
      vels = data[:, 3:6]
      trial_vels.append(vels)
      accels_vec = np.diff(vels, axis=0) / dt
      accels_vec = np.vstack((accels_vec, np.zeros(3)))
      trial_rpys.append(data[:, 6:9])
      trial_omegas.append(data[:, 9:12])

    elif '2d' in args.system:
      poss_vec = data[:, :2]
      vels = data[:, 2:4]
      trial_vels.append(vels)
      accels_vec = np.diff(vels, axis=0) / dt
      accels_vec = np.vstack((accels_vec, np.zeros(2)))
      trial_rpys.append(data[:, 4:5])
      trial_omegas.append(data[:, 5:6])

    elif args.system in ['linear', 'linearpos', 'nl1d']:
      poss_vec = data[:, 0:1]
      vels = data[:, 1:2]
      trial_vels.append(vels)
      accels_vec = np.diff(vels, axis=0) / dt
      accels_vec = np.vstack((accels_vec, 0))
      trial_rpys.append(data[:, 2:3])
      trial_omegas.append(data[:, 3:4])

    elif args.system == 'simple':
      poss_vec = data[:, 0:1]
      accels_vec = np.diff(data[:, 1:2], axis=0) / dt
      accels_vec = np.vstack((accels_vec, 0))

    elif args.system == 'trivial':
      poss_vec = data[:, 0:1]
      accels_vec = np.diff(data[:, 0:1], axis=0) / dt
      accels_vec = np.vstack((accels_vec, 0))

    pos_errors = poss_vec - poss_des_vec
    abs_pos_errors = np.abs(pos_errors)
    accel_errors = accels_vec - accels_des_vec
    abs_accel_errors = np.abs(accel_errors)

    title_s = "Iteration %d" % (iter_no + 1)
    print("============")
    print(title_s)
    print("============")
    print("Avg. pos error:", np.mean(abs_pos_errors))
    print("Avg. Y   error:", np.mean(abs_pos_errors[:, AXIS]))
    print("Max. pos error:", np.max(abs_pos_errors))
    #print("Avg. acc error:", np.mean(abs_accel_errors))
    #print("Max. acc error:", np.max(abs_accel_errors))

    trial_poss.append(poss_vec)
    trial_accels.append(accels_vec)

    trial_controls.append(np.array(controller.final_controls))
    trial_control_corrections.append(lifted_control.copy())

    if iter_no >= args.trials - 1:
      break

    states = []
    controls = []
    for i in list(range(N)) + [N - 1]:
      if args.relin_time:
        ind = i
      else:
        ind = 0

      if args.relin_iter:
        state = data[ind, :]
        control = lifted_control[ilc.n_control * ind : ilc.n_control * (ind + 1)]
      else:
        state = ff_states[ind]
        control = initial_lifted_control[ilc.n_control * ind : ilc.n_control * (ind + 1)]

      states.append(state)
      controls.append(control)

    calCBpD = ilc.get_learning_operator(dt, states, controls, poss_des_vec, vels_des_vec, accels_des_vec, jerks_des_vec, snaps_des_vec)

    if args.noise:
      for i in range(len(pos_errors)):
        pos_errors[i] += np.random.normal(0, args.noise_stddev)

    lifted_output_error = np.zeros((ilc.n_out * N))
    for i in range(N):
      if 'pos' in args.system or '3d' in args.system:
        if args.filter:
          for j in range(DIMS):
            pos_errors[:, j] = savgol_filter(pos_errors[:, j], window_length=11, polyorder=3)

        lifted_output_error[ilc.n_out * i : ilc.n_out * (i + 1)] = pos_errors[i + 1]
      else:
        lifted_output_error[ilc.n_out * i : ilc.n_out * (i + 1)] = accel_errors[i + 1]

    # ILC update
    # Fu = y => arg min (u)  || Fu - y ||
    # Want: arg min (u) || Fu - y || + alpha || u ||
    min_norm_mat = np.diag(np.tile(ilc.control_normalization, N))
    F = np.vstack((calCBpD, args.w * min_norm_mat))
    y = np.hstack((lifted_output_error, np.zeros(calCBpD.shape[1])))
    update, _, _, _ = np.linalg.lstsq(F, -y, rcond=None)

    update *= args.alpha

    lifted_control += update
    cum_updates += update

    if args.plot_updates:
      plt.figure()
      for i in range(ilc.n_control):
        plt.subplot(ilc.n_control, 1, i + 1)
        if not i: plt.title("ILC Control Updates")
        plt.plot(ts[:-1], update[i::ilc.n_control])
        plt.ylabel('Control %d' % (i + 1))
        plt.xlabel("Time (s)")

      plt.figure()
      for i in range(ilc.n_control):
        plt.subplot(ilc.n_control, 1, i + 1)
        if not i: plt.title("ILC Cumulative Control Updates")
        plt.plot(ts[:-1], cum_updates[i::ilc.n_control])
        plt.ylabel('Control %d' % (i + 1))
        plt.xlabel("Time (s)")

      plt.show()

  start_color = np.array((1, 0, 0, 0.5))
  end_color = np.array((0, 1, 0, 0.5))

  def plot_trials(datas, desired, title, ylabel):
    axes = "XYZ" if DIMS == 3 else "XZ" if DIMS == 2 else "X"
    #for axis in [AXIS]:
    for axis in range(datas[0].shape[1]):
      for i, trial_data in enumerate(datas):
        if np.linalg.norm(trial_data[:, axis]) > 1e-8:
          break
      else:
        continue

      title_s = "Actual vs. Desired %s %s" % (title, axes[axis])
      plt.figure(title_s)
      if desired is not None:
        plt.plot(ts, desired[:, axis], "k:", linewidth=2, label="Desired")

      for i, trial_data in enumerate(datas):
        alpha = float(i) / len(datas)
        line_color = (1 - alpha) * start_color + alpha * end_color

        plot_args = {}
        if i == 0 or i == len(datas) - 1:
          plot_args['label'] = "Trial %d" % (i + 1)

        plt.plot(ts[:len(trial_data)], trial_data[:, axis], color=line_color, linewidth=2, **plot_args)

      plt.xlabel("Time (s)")
      plt.ylabel(ylabel % axes[axis])
      plt.legend()
      plt.title(title_s)

    #plt.savefig("ilc_%s.png" % title.lower())

  if args.save:
    import json
    import os
    import time
    timepath = time.strftime("%Y%m%d-%H%M%S")
    dirname = os.path.join("data", timepath)
    os.mkdir(dirname)

    param_file = open(os.path.join(dirname, "params.txt"), 'w')
    param_file.write(str(vars(args)))
    param_file.close()

    for i in range(len(trial_poss)):
      suffix = "%02d.txt" % i
      np.savetxt(os.path.join(dirname, "pos" + suffix), trial_poss[i], delimiter=',')

      if '3d' in args.system or '2d' in args.system:
        np.savetxt(os.path.join(dirname, "rpy" + suffix), trial_rpys[i], delimiter=',')
        np.savetxt(os.path.join(dirname, "angvel" + suffix), trial_omegas[i], delimiter=',')

      np.savetxt(os.path.join(dirname, "control-corrections" + suffix), trial_control_corrections[i], delimiter=',')
      np.savetxt(os.path.join(dirname, "controls" + suffix), trial_controls[i], delimiter=',')

    if args.feedback:
      resp = np.array((controller.feedback_responses))
      for i in range(resp.shape[1]):
        np.savetxt(os.path.join(dirname, "fbresp%01d.txt" % i), resp[:, i, :], delimiter=',')

    latest_dir = os.path.join("data", "latest")
    if os.path.exists(latest_dir):
      os.remove(latest_dir)

    os.symlink(timepath, latest_dir)
    print("Data written to %s (latest)" % dirname)

  if args.plot:
    plot_trials(trial_poss, poss_des_vec, "Position", "Pos. %s (m)")
    #plot_trials(trial_vels, vels_des_vec, "Velocity", "Vel. %s (m/s)")
    #plot_trials(trial_accels, accels_des_vec, "Acceleration", "Accel. %s (m/s^2)")

    if '3d' in args.system or '2d' in args.system:
      plot_trials(trial_omegas, None, "Angular Velocity", "$\omega$ %s (rad/s)")
      plot_trials(trial_rpys, None, "Angle", "$\\alpha$ %s (rad/s^2)")

  if args.plot_control_corrections:
    for j in range(ilc.n_control):
      for i, trial_data in enumerate(trial_control_corrections):
        if np.linalg.norm(trial_data[j::ilc.n_control]) > 1e-8:
          break
      else:
        continue

      title_s = "Control Corrections (%d)" % j
      plt.figure(title_s)

      for i, trial_data in enumerate(trial_control_corrections):
        alpha = float(i) / len(trial_control_corrections)
        line_color = (1 - alpha) * start_color + alpha * end_color

        plot_args = {}
        if i == 0 or i == len(trial_control_corrections) - 1:
          plot_args['label'] = "Trial %d" % (i + 1)

        plt.plot(ts[:-1], trial_data[j::ilc.n_control], color=line_color, linewidth=2, **plot_args)

      plt.xlabel("Time (s)")
      plt.ylabel(title_s)
      plt.legend()
      plt.title(title_s)

  if args.plot_controls:
    for j in range(ilc.n_control):
      for i, trial_data in enumerate(trial_controls):
        if np.linalg.norm(trial_data[:, j]) > 1e-8:
          break
      else:
        continue

      title_s = "Controls (%d)" % j
      plt.figure(title_s)

      for i, trial_data in enumerate(trial_controls):
        alpha = float(i) / len(trial_controls)
        line_color = (1 - alpha) * start_color + alpha * end_color

        plot_args = {}
        if i == 0 or i == len(trial_controls) - 1:
          plot_args['label'] = "Trial %d" % (i + 1)

        plt.plot(ts[:-1], trial_data[:, j], color=line_color, linewidth=2, **plot_args)

      plt.xlabel("Time (s)")
      plt.ylabel(title_s)
      plt.legend()
      plt.title(title_s)

  if args.plot_fb_resp:
    resp = np.array((controller.feedback_responses))
    resp_ana = np.array((controller.feedback_responses_ana))

    for i, tit in enumerate(ilc.state_labels):
      for j, ctit in enumerate(ilc.control_labels):

        if np.linalg.norm(resp[:, j, i]) < 1e-4 and (not len(resp_ana) or np.linalg.norm(resp_ana[:, j, i]) < 1e-6):
          continue

        plt.figure()
        plt.plot(ts[:-1], resp[:, j, i], label='d%d / d%d' % (j, i))
        if len(resp_ana):
          plt.plot(ts[:-1], resp_ana[:, j, i], label='d%d / d%d (ana)' % (j, i))

          if not np.allclose(resp_ana[:, j, i], resp[:, j, i]):
            err = np.mean(np.abs(resp_ana[:, j, i] - resp[:, j, i]))
            print("ERROR: FB resp for", ctit, "/", tit, "doesn't match! Avg. error is", err)

        plt.title("d %s / d %s" % (ctit, tit))
        plt.legend()

  plt.show()
