from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from ilc_models import base, trivial, one, quadlin, quadlinpos, nl1d, quad2dlin, quad2d, quad2ddedi, quad2ddedis, quad3d, quad3dtv, quad3dfl, quad3dflv, quad3dfltd, quad3dfls
from python_utils.polyu import deriv_fitting_matrix


def get_poly(x, v=0, a=0, j=0, end_pos=1.0, duration=1.0):
  poly_fit_mat = np.linalg.inv(deriv_fitting_matrix(8, t_end=duration))
  poly = poly_fit_mat.dot(np.array((x, v, a, j, end_pos, 0, 0, 0)))
  return poly[::-1]

system_map = {
               # ilc, DIMS
  'trivial':   (trivial.Trivial, 1),
  'simple':    (one.One, 1),
  'linear':    (quadlin.QuadLin, 1),
  'linearpos': (quadlinpos.QuadLinPos, 1),
  'nl1d' :     (nl1d.NL1D, 1),
  '2dposlin':  (quad2dlin.Quad2DLin, 2),
  '2dpos':     (quad2d.Quad2D, 2),
  '3d':        (quad3d.Quad3D, 3),
  '3dtv':      (quad3dtv.Quad3DTV, 3),
  '2ddedi':    (quad2ddedi.Quad2DDEDI, 2),
  '2ddedis':   (quad2ddedis.Quad2DDEDIS, 2),
  '3ddedi':    (quad3dfl.Quad3DFL, 3),
  '3ddediv':   (quad3dflv.Quad3DFLV, 3),
  '3ddeditd':  (quad3dfltd.Quad3DFLTD, 3),
  '3ddedis':   (quad3dfls.Quad3DFLS, 3),
}

def get_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # General
  parser.add_argument("--system", type=str, default="simple", choices=system_map.keys(), help="Type of system to simulate.")
  parser.add_argument("--sim-dt", type=float, default=0.02, help="Time between simulation steps.")

  # Trajectory
  parser.add_argument("--dist", default=1.5, type=float, help="Distance to travel.")
  parser.add_argument("--traj-duration", default=1.0, type=float, help="Trajectory duration.")
  parser.add_argument("--rest-time", default=0.0, type=float, help="Rest time to add to end of desired trajectory.")
  parser.add_argument("--feedback", "--fb", default=False, action='store_true', help="Apply feedback along the trajectory.")
  parser.add_argument("--feedforward", "--ff", default=False, action='store_true', help="Use initial feedforward trajectory.")
  parser.add_argument("--step", default=False, action='store_true', help="Execute a step.")

  # Controllers
  #   Quad3D
  parser.add_argument("--cascaded-thrust", default="project", choices=["norm", "project", "maintain-z"], type=str, help="How to calculate the thrust control input in the cascaded PD Quad3D feedback controller.")

  # ILC
  parser.add_argument("--ilc-dt", type=float, default=0.02, help="Time between ILC corrections.")
  parser.add_argument("--trials", type=int, default=4, help="Number of ILC trials to run.")
  parser.add_argument("--alpha", type=float, default=1.0, help="Percentage of update (0 - 1) to use at each iteration. Lower values increase stability.")
  parser.add_argument("--relin-time", default=True, action='store_true', help="Use a different linearization point at each time step along the trajectory.")
  parser.add_argument("--relin-iter", default=True, action='store_true', help="Use different linearization points for each iteration.")
  parser.add_argument("--no-relin-time", default=False, dest='relin_time', action='store_false')
  parser.add_argument("--no-relin-iter", default=False, dest='relin_iter', action='store_false')
  parser.add_argument("--w", default=1e-1, type=float, help="Weight of control update norm minimization.")
  parser.add_argument("--filter", default=False, action='store_true', help="Filter the position errors fed into ILC.")

  parser.add_argument("--check-fb-resp", default=False, action='store_true', help="Check the feedback response along the final trajectory against numerical differentiation.")

  # Disturbances
  parser.add_argument("--thrust-dist", default=1.0, type=float, help="Disturbance used to scale the commanded thrust u.")
  parser.add_argument("--drag-dist", default=0.0, type=float, help="Disturbance used to scale velocity subtracted from acceleration as drag.")
  parser.add_argument("--model-drag", default=False, action='store_true', help="Consider the drag disturbance explicitly in the model.")
  parser.add_argument("--periodic-accel-dist-mag", default=0.0, type=float, help="Magnitude of periodic accel dist.")
  parser.add_argument("--periodic-accel-dist-periods", default=1.0, type=float, help="No. of periods of periodic accel dist.")
  # TODO: Below currently only supported by Quad3D.
  parser.add_argument("--angaccel-dist", default=1.0, type=float, help="Disturbance used to scale the commanded angular acceleration alpha.")
  parser.add_argument("--delay-control", default=False, action='store_true', help="Delay the control inputs applied to the real system.")
  parser.add_argument("--delay-timeconstant", default=20.0, type=float, help="Time constant used to delay the control inputs.")
  parser.add_argument("--delay-timeconstant-control", default=20.0, type=float, help="Time constant assumed in the controller (quad3dfltd).")
  parser.add_argument("--limit-motor", default=False, action='store_true', help="Limit force output of a motor.")
  parser.add_argument("--limit-motor-ind", default=0, type=int, help="Index of motor to limit force output of.")
  parser.add_argument("--limit-motor-scale", default=0.75, type=float, help="Amount to scale force output of motor by.")
  parser.add_argument("--positive-thrust-only", default=True, action='store_true')
  parser.add_argument("--allow-negative-thrust", default=False, action='store_false', dest='positive_thrust_only', help="Allow negative thrust in 3D Quads.")
  parser.add_argument("--angaccel-limit", default=3000, type=float, help="Maximum magnitude of the angular acceleration control input for 3D Quads.")
  parser.add_argument("--accel-limit", default=50, type=float, help="Maximum magnitude of the linear acceleration control input for 3D Quads.")

  parser.add_argument("--noise", default=False, action='store_true', help="Add noise to the position errors fed into ILC.")
  parser.add_argument("--noise-stddev", default=1e-3, type=float, help="Stddev of noise added to the position errors")

  parser.add_argument("--poke", default=False, action='store_true', help="Add a \"poke\" disturbance during the 3D trajectory.")
  parser.add_argument("--poke-strength", default=600, type=float, help="Value of constant poke angular acceleration disturbance.")
  parser.add_argument("--poke-time", default=0.5, type=float, help="Time of poke.")
  parser.add_argument("--poke-duration", default=0.03, type=float, help="Duration of poke.")

  # Output Options
  parser.add_argument("--no-stdout", default=False, action='store_true', help="Print stats to stdout.")
  parser.add_argument("--print-params", default=False, action='store_true', help="Print tabulated params at start.")
  parser.add_argument("--plot", default=False, action='store_true', help="Plot the states for each trial.")
  parser.add_argument("--plot-fb-resp", default=False, action='store_true', help="Plot the feedback response along the final trajectory.")
  parser.add_argument("--plot-controls", default=False, action='store_true', help="Plot the final control inputs used.")
  parser.add_argument("--plot-control-corrections", default=False, action='store_true', help="Plot the final ILC control corrections used.")
  parser.add_argument("--plot-updates", default=False, action='store_true', help="Plot the ILC updates after every iteration.")
  parser.add_argument("--plot-all", default=False, action='store_true', help="Do not skip plotting trials that are all zero.")

  # Save Options
  parser.add_argument("--save", default=False, action='store_true', help="Write parameters and trajectories to file.")
  parser.add_argument("--save-dir-prefix", default="", type=str, help="Prefix for the output directory before the timestamp.")
  parser.add_argument("--save-symlink", default="latest", type=str, help="Symlink to the output directory")

  return parser

class ILCExperiment(object):
  def __init__(self, args):
    assert args.relin_time or not args.relin_iter
    assert not (args.relin_time and (not args.relin_iter) and (not args.feedforward))

    compute_fb_resp = args.plot_fb_resp or args.check_fb_resp

    assert args.feedback or not compute_fb_resp

    ilc_c, DIMS = system_map[args.system]
    ilc = ilc_c(**vars(args))
    AXIS = 1 if DIMS == 3 else 0

    ilc_dt = args.ilc_dt
    sim_dt = args.sim_dt
    poly_tend = args.traj_duration
    t_end = poly_tend + args.rest_time

    poly_N = int(round(poly_tend / sim_dt))
    poly_ts = np.linspace(0, poly_tend, poly_N + 1)

    N = int(round(t_end / sim_dt))
    ts = np.linspace(0, t_end, N + 1)

    N_ilc = int(round(t_end / ilc_dt))
    ts_ilc = np.linspace(0, t_end, N_ilc + 1)

    poke_center = args.poke_time / sim_dt
    poke_steps = args.poke_duration / sim_dt

    if not args.no_stdout:
      print("No. of ILC steps is", N_ilc)
      print("No. of sim steps is", N)

    def pad_des(des, val=0):
      """ Returns a desired vectory spanning the full length of time
          i.e. adds the rest time in addition to the poly traj time """
      return np.hstack((des, val * np.ones(N - poly_N)))

    if not args.step:
      pos_poly = get_poly(0, end_pos=args.dist, duration=poly_tend)
      vel_poly = np.polyder(pos_poly)
      acc_poly = np.polyder(vel_poly)
      jerk_poly = np.polyder(acc_poly)
      snap_poly = np.polyder(jerk_poly)

      poss_des = np.polyval(pos_poly, poly_ts)
      poss_des = pad_des(poss_des, args.dist)

      vels_des = np.polyval(vel_poly, poly_ts)
      vels_des = pad_des(vels_des)

      accels_des = pad_des(np.polyval(acc_poly, poly_ts))
      jerks_des = pad_des(np.polyval(jerk_poly, poly_ts))
      snaps_des = pad_des(np.polyval(snap_poly, poly_ts))
    else:
      poss_des = pad_des(args.dist * np.ones(len(poly_ts)), args.dist)
      vels_des = pad_des(np.zeros(len(poly_ts)))
      accels_des = pad_des(np.zeros(len(poly_ts)))
      jerks_des = pad_des(np.zeros(len(poly_ts)))
      snaps_des = pad_des(np.zeros(len(poly_ts)))

    lifted_control = np.zeros(N_ilc * ilc.n_control)
    if (DIMS == 2 or DIMS == 3) and not args.feedback:
      lifted_control[::ilc.n_control] = base.g

    cum_updates = np.zeros(lifted_control.shape)

    if args.feedforward:
      assert hasattr(ilc, 'feedforward')

      ff_states = []
      ff_controls = []

      for i in range(N):
        hods = [poss_des[i], vels_des[i], accels_des[i], jerks_des[i], snaps_des[i]]
        hod_vecs = []
        for hod in hods:
          vec = np.zeros(DIMS)
          vec[AXIS] = hod
          hod_vecs.append(vec)

        state, control = ilc.feedforward(*hod_vecs)
        ff_states.append(state)
        ff_controls.append(control)

      ff_states = np.array(ff_states)
      ff_controls = np.array(ff_controls)

      if not args.feedback:
        ff_controls_interp = interp1d(ts[:-1], ff_controls, axis=0)(ts_ilc[:-1])
        for i in range(N_ilc):
          lifted_control[ilc.n_control * i : ilc.n_control * (i + 1)] = ff_controls_interp[i, :]

    initial_lifted_control = lifted_control.copy()

    # Do we need this? TODO
    # I had originally added this to work well with low N simulations.
    # i.e. achieve perfect error.
    #if args.system in ['linearpos', 'nl1d'] or '3d' in args.system or '2d' in args.system:
    #  for i in range(min(4, N)):
    #    poss_des[i] = 0.0

    #if args.system in ['linear', '2d', 'linearpos']:
    #  accels_des[1] = 0.0

    poss_des[0] = 0.0
    poss_des[1] = 0.0

    poss_des_vec = np.zeros((N + 1, DIMS))
    poss_des_vec[:, AXIS] = poss_des

    vels_des_vec = np.zeros((N + 1, DIMS))
    vels_des_vec[:, AXIS] = vels_des

    accels_des_vec = np.zeros((N + 1, DIMS))
    accels_des_vec[:, AXIS] = accels_des

    jerks_des_vec = np.zeros((N + 1, DIMS))
    jerks_des_vec[:, AXIS] = jerks_des

    snaps_des_vec = np.zeros((N + 1, DIMS))
    snaps_des_vec[:, AXIS] = snaps_des

    poss_des_interp = interp1d(ts, poss_des_vec, axis=0)(ts_ilc)
    vels_des_interp = interp1d(ts, vels_des_vec, axis=0)(ts_ilc)
    accels_des_interp = interp1d(ts, accels_des_vec, axis=0)(ts_ilc)
    jerks_des_interp = interp1d(ts, jerks_des_vec, axis=0)(ts_ilc)
    snaps_des_interp = interp1d(ts, snaps_des_vec, axis=0)(ts_ilc)

    class Controller:
      def __init__(self, lifted_control, poss_des, vels_des, accels_des, jerks_des, snaps_des):
        self.poss_des = poss_des
        self.vels_des = vels_des

        self.accs_des = []
        self.jerks_des = []
        self.snaps_des = []
        self.angvels_des = []
        self.angaccels_des = []

        controls_ilc = np.zeros((N_ilc, ilc.n_control))
        for i in range(controls_ilc.shape[0]):
          controls_ilc[i, :] = lifted_control[ilc.n_control * i : ilc.n_control * (i + 1)]

        self.controls = interp1d(ts_ilc[:-1], controls_ilc, axis=0, fill_value="extrapolate")(ts[:-1])

        for i in range(N):
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
            self.angvels_des.append(self.jerks_des[-1])
            self.angaccels_des.append(self.snaps_des[-1])

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
          else:
              self.angvels_des.append(0)
              self.angaccels_des.append(0)

        self.index = 0

        self.compute_feedback_response = False
        self.feedback_responses = []
        self.feedback_responses_ana = []
        self.final_controls = []

      def get(self, x):
        ilc_controls = self.controls[self.index]
        if args.feedback:
          kwargs = dict(
            x=x,
            dt=sim_dt,
            pos_des=self.poss_des[self.index],
            vel_des=self.vels_des[self.index],
            acc_des=self.accs_des[self.index],
            jerk_des=self.jerks_des[self.index],
            snap_des=self.snaps_des[self.index],
            u_ilc=ilc_controls,
            angvel_des=self.angvels_des[self.index],
            angaccel_des=self.angaccels_des[self.index],
          )

          if self.compute_feedback_response:
            feedback = ilc.feedback(integrate=False, **kwargs)

            if args.system in ['3ddediv', '3ddeditd']:
              if args.system == '3ddediv':
                feedback[0] = ilc.v1
              else:
                feedback[0] = ilc.int_udot

              init_snap = ilc.snap
              init_acc = ilc.acc
              init_z = ilc.z

            # Hack TODO XXX To deal with systems where the ILC corrections
            # are not equal to true system control inputs.
            if hasattr(ilc, 'n_control_sys'):
              response = np.zeros((ilc.n_control_sys, ilc.n_state))
            else:
              response = np.zeros((ilc.n_control, ilc.n_state))

            try_eps = 1e-6
            for i in range(len(x)):
              x_try = x.copy()
              x_try[i] += try_eps

              kwargs['x'] = x_try
              feedback_try = ilc.feedback(integrate=False, **kwargs)

              #if i == 6:
              #  print(ilc.snap, ilc.z)
              #  print((ilc.snap - init_snap) / try_eps)
              #  #print((ilc.acc - init_acc) / try_eps)
              #  #print((ilc.z - init_z) / try_eps)

              if args.system in ['3ddediv', '3ddedis']:
                feedback_try[0] = ilc.v1
              if args.system == '3ddeditd':
                feedback_try[0] = ilc.int_udot

              deriv = (feedback_try - feedback) / try_eps

              response[:, i] = deriv

            if args.system == '3ddediv':
              kwargs['x'] = x
              orig_u, orig_udot = ilc.int_u, ilc.int_udot
              ilc.int_u = orig_u + try_eps

              feedback_try = ilc.feedback(integrate=False, **kwargs)
              feedback_try[0] = ilc.v1

              deriv = (feedback_try - feedback) / try_eps
              response[:, 12] = deriv

              ilc.int_u = orig_u
              ilc.int_udot = orig_udot + try_eps

              feedback_try = ilc.feedback(integrate=False, **kwargs)
              feedback_try[0] = ilc.v1

              deriv = (feedback_try - feedback) / try_eps
              response[:, 13] = deriv

              ilc.int_u = orig_u
              ilc.int_udot = orig_udot

            #if args.system == '3ddeditd':
            #  kwargs['x'] = x
            #  orig_u, orig_c = ilc.int_u, ilc.int_c
            #  ilc.int_u = orig_u + try_eps

            #  feedback_try = ilc.feedback(**kwargs, integrate=False)
            #  feedback_try[0] = ilc.int_udot

            #  deriv = (feedback_try - feedback) / try_eps
            #  response[:, 12] = deriv

            #  ilc.int_u = orig_u
            #  ilc.int_c = orig_c + try_eps

            #  feedback_try = ilc.feedback(**kwargs, integrate=False)
            #  feedback_try[0] = ilc.int_udot

            #  deriv = (feedback_try - feedback) / try_eps
            #  response[:, 13] = deriv

            #  ilc.int_u = orig_u
            #  ilc.int_c = orig_c

            self.feedback_responses.append(response)
            ilc.pos_des = self.poss_des[self.index]
            ilc.vel_des = self.vels_des[self.index]
            ilc.acc_des = self.accs_des[self.index]
            ilc.jerk_des = self.jerks_des[self.index]
            ilc.snap_des = self.snaps_des[self.index]
            if hasattr(ilc, "get_feedback_response"):
              self.feedback_responses_ana.append(ilc.get_feedback_response(x, ilc_controls, sim_dt)[0])

          kwargs['x'] = x
          #x[6] += np.random.normal(scale=0.0005)
          #if self.compute_feedback_response and args.poke:
            #if 100 < self.index < 150:
          #    kwargs['x'][6] += 0.1 * np.random.normal()

          feedback = ilc.feedback(**kwargs)

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

    cached_pinv = None

    for iter_no in range(args.trials):
      controller = Controller(lifted_control, poss_des_vec, vels_des_vec, accels_des_vec, jerks_des_vec, snaps_des_vec)
      controller.compute_feedback_response = iter_no == args.trials - 1 and args.feedback and (compute_fb_resp)# or args.save)
      controller.poke = iter_no == args.trials - 1 and args.poke
      ilc.reset()
      data = ilc.simulate(t_end, controller.get, dt=sim_dt)

      if '3d' in args.system:
        poss_vec = data[:, :3]
        vels = data[:, 3:6]
        trial_vels.append(vels)
        accels_vec = np.diff(vels, axis=0) / sim_dt
        accels_vec = np.vstack((accels_vec, np.zeros(3)))
        trial_rpys.append(data[:, 6:9])
        trial_omegas.append(data[:, 9:12])

      elif '2d' in args.system:
        poss_vec = data[:, :2]
        vels = data[:, 2:4]
        trial_vels.append(vels)
        accels_vec = np.diff(vels, axis=0) / sim_dt
        accels_vec = np.vstack((accels_vec, np.zeros(2)))
        trial_rpys.append(data[:, 4:5])
        trial_omegas.append(data[:, 5:6])

      elif args.system in ['linear', 'linearpos', 'nl1d']:
        poss_vec = data[:, 0:1]
        vels = data[:, 1:2]
        trial_vels.append(vels)
        accels_vec = np.diff(vels, axis=0) / sim_dt
        accels_vec = np.vstack((accels_vec, 0))
        trial_rpys.append(data[:, 2:3])
        trial_omegas.append(data[:, 3:4])

      elif args.system == 'simple':
        poss_vec = data[:, 0:1]
        accels_vec = np.diff(data[:, 1:2], axis=0) / sim_dt
        accels_vec = np.vstack((accels_vec, 0))

      elif args.system == 'trivial':
        poss_vec = data[:, 0:1]
        accels_vec = np.diff(data[:, 0:1], axis=0) / sim_dt
        accels_vec = np.vstack((accels_vec, 0))

      pos_errors = poss_vec - poss_des_vec
      poserr_norms = np.linalg.norm(pos_errors, axis=1)
      abs_pos_errors = np.abs(pos_errors)
      accel_errors = accels_vec - accels_des_vec
      abs_accel_errors = np.abs(accel_errors)

      if not args.no_stdout:
        title_s = "Iteration %d" % (iter_no + 1)
        print("============")
        print(title_s)
        print("============")
        print("Avg. pos error:", np.mean(poserr_norms))
        print("Avg. Y   error:", np.mean(abs_pos_errors[:, AXIS]))
        if DIMS > 1:
          print("Avg. Z   error:", np.mean(abs_pos_errors[:, AXIS + 1]))
          if args.step:
            print("Max. Z   error:", np.max(abs_pos_errors[:, AXIS + 1]))
        print("Max. pos error:", np.max(poserr_norms))
        #print("Avg. acc error:", np.mean(abs_accel_errors))
        #print("Max. acc error:", np.max(abs_accel_errors))

      trial_poss.append(poss_vec)
      trial_accels.append(accels_vec)

      trial_controls.append(np.array(controller.final_controls))
      trial_control_corrections.append(lifted_control.copy())

      if iter_no >= args.trials - 1:
        break

      if args.relin_iter:
        data_interp = interp1d(ts, data, axis=0)(ts_ilc)
      else:
        ff_states_interp = interp1d(ts, fs_states, axis=0)(ts_ilc)

      states = []
      controls = []
      for i in list(range(N_ilc)) + [N_ilc - 1]:
        if args.relin_time:
          ind = i
        else:
          ind = 0

        if args.relin_iter:
          state = data_interp[ind, :]
          control = lifted_control[ilc.n_control * ind : ilc.n_control * (ind + 1)]
        else:
          state = ff_states_interp[ind]
          control = initial_lifted_control[ilc.n_control * ind : ilc.n_control * (ind + 1)]

        states.append(state)
        controls.append(control)

      if args.noise:
        for i in range(len(pos_errors)):
          pos_errors[i] += np.random.normal(0, args.noise_stddev)

      if args.filter:
        for j in range(DIMS):
          pos_errors[:, j] = savgol_filter(pos_errors[:, j], window_length=11, polyorder=3)

      pos_errors_interp = interp1d(ts, pos_errors, axis=0)(ts_ilc)

      if N_ilc == N:
        assert(np.allclose(pos_errors, pos_errors_interp))

      lifted_output_error = np.zeros((ilc.n_out * N_ilc))
      for i in range(N_ilc):
        if 'pos' in args.system or '3d' in args.system or '2d' in args.system or args.system in ['trivial', 'simple']:
          lifted_output_error[ilc.n_out * i : ilc.n_out * (i + 1)] = pos_errors_interp[i + 1]
        else:
          lifted_output_error[ilc.n_out * i : ilc.n_out * (i + 1)] = accel_errors[i + 1]

      #err_xs = np.linspace(0, 1, N_ilc)
      #err_xs = np.ones(N_ilc)
      #err_xs[:N_ilc//4] = 0
      #err_scale1 = 1 / (1 + (err_xs / (1 - err_xs)) ** (-2))
      #error_scale = np.repeat(err_xs, ilc.n_out)
      #lifted_output_error *= error_scale

      #senss = []
      #inv = np.linalg.pinv(F)
      #errs = np.random.normal(0, 0.01, size=(inv.shape[1],))
      #res = inv.dot(errs)
      #u1 = res[0::ilc.n_control]
      #u2 = res[1::ilc.n_control]
      #u3 = res[2::ilc.n_control]
      #u4 = res[3::ilc.n_control]
      #print(inv.shape)
      #for i in range(len(inv) // 4):
        #senss.append(np.linalg.norm(inv[4 * i + 0, :]))

      #plt.figure()
      #plt.plot(u1)
      #plt.figure()
      #plt.plot(u2)
      #plt.figure()
      #plt.plot(u3)
      #plt.figure()
      #plt.plot(u4)
      #plt.show()

      y = np.hstack((lifted_output_error, np.zeros(N_ilc * ilc.n_control)))

      if not ilc.constant_ilc_mats or cached_pinv is None:
        # ILC update
        # Fu = y => arg min (u)  || Fu - y ||
        # Want: arg min (u) || Fu - y || + alpha || u ||
        min_norm_mat = np.diag(np.tile(ilc.control_normalization, N_ilc))
        calCBpD, G = ilc.get_learning_operator(ilc_dt, states, controls, poss_des_interp, vels_des_interp, accels_des_interp, jerks_des_interp, snaps_des_interp)

        #if args.feedback:
        #  min_norm_mat = min_norm_mat.dot(G)

        F = np.vstack((calCBpD, args.w * min_norm_mat))

        if ilc.constant_ilc_mats:
          cached_pinv = np.linalg.pinv(F)
          #print(cached_pinv[:20, :20])
          #print(np.count_nonzero(cached_pinv))
          #print(np.count_nonzero(cached_pinv))

          #cached_pinv2 = np.linalg.pinv(calCBpD)

          #disp = np.ones(cached_pinv2.shape)
          #disp[np.abs(cached_pinv2) < 1e-3] = 0.0

          #plt.figure()
          #plt.imshow(calCBpD[::3, ::3], 'gray')
          #plt.figure()
          #plt.imshow(cached_pinv2[::3, ::3], 'gray')
          #plt.figure()
          #plt.imshow(disp[::3, ::3], 'gray')
          #print(cached_pinv[::3, ::3].shape[0] * cached_pinv[::3, ::3].shape[1])
          #print(np.count_nonzero(F[::3, ::3]))
          #print(np.count_nonzero(cached_pinv[::3, ::3]))
          #print(np.diag(cached_pinv2))
          #plt.show()
        else:
          update, _, _, _ = np.linalg.lstsq(F, -y, rcond=None)

      if cached_pinv is not None:
        update = cached_pinv.dot(-y)

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
      all_skipped = True
      axes = "XYZ" if DIMS == 3 else "XZ" if DIMS == 2 else "X"
      #for axis in [AXIS]:
      for axis in range(datas[0].shape[1]):
        if not args.plot_all:
          for i, trial_data in enumerate(datas):
            if np.linalg.norm(trial_data[:, axis]) > 1e-8:
              break
          else:
            continue

        all_skipped = False

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

      if all_skipped:
        print("WARNING: All trials skipped plotting because nothing interesting happened (pass --plot-all).")
      #plt.savefig("ilc_%s.png" % title.lower())

    if args.save:
      import json
      import os
      import time
      timepath = time.strftime("%Y%m%d-%H%M%S")
      dir_leafname = args.save_dir_prefix + "-" + timepath
      dirname = os.path.join("data", dir_leafname)
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

      #if args.feedback:
      #  resp = np.array((controller.feedback_responses))
      #  for i in range(resp.shape[1]):
      #    np.savetxt(os.path.join(dirname, "fbresp%01d.txt" % i), resp[:, i, :], delimiter=',')

      symlink_dir = os.path.join("data", args.save_symlink)
      if os.path.exists(symlink_dir):
        os.remove(symlink_dir)

      os.symlink(dir_leafname, symlink_dir)
      print("Data written to %s (%s)" % (dirname, args.save_symlink))

    if args.plot:
      plot_trials(trial_poss, poss_des_vec, "Position", "Pos. %s (m)")
      #plot_trials(trial_vels, vels_des_vec, "Velocity", "Vel. %s (m/s)")
      #plot_trials(trial_accels, accels_des_vec, "Acceleration", "Accel. %s (m/s^2)")

      if '3d' in args.system or '2d' in args.system:
        #plot_trials(trial_omegas, None, "Angular Velocity", "$\omega$ %s (rad/s)")
        plot_trials(trial_rpys, None, "Angle", "$\\alpha$ %s (rad/s^2)")

    if args.plot_control_corrections:
      for j in range(ilc.n_control):
        for i, trial_data in enumerate(trial_control_corrections):
          if np.linalg.norm(trial_data[j::ilc.n_control]) > 1e-8:
            break
        else:
          continue

        title_s = "Control Corrections (%s)" % ilc.control_labels[j]
        plt.figure(title_s)

        for i, trial_data in enumerate(trial_control_corrections):
          alpha = float(i) / len(trial_control_corrections)
          line_color = (1 - alpha) * start_color + alpha * end_color

          plot_args = {}
          if i == 0 or i == len(trial_control_corrections) - 1:
            plot_args['label'] = "Trial %d" % (i + 1)

          plt.plot(ts_ilc[:-1], trial_data[j::ilc.n_control], color=line_color, linewidth=2, **plot_args)

        plt.xlabel("Time (s)")
        plt.ylabel(title_s)
        plt.legend()
        plt.title(title_s)

    if args.plot_controls:
      for j in range(ilc.n_control_sys):
        for i, trial_data in enumerate(trial_controls):
          if np.linalg.norm(trial_data[:, j]) > 1e-8:
            break
        else:
          continue

        title_s = "Controls (%s)" % ilc.sys_control_labels[j]
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

    if compute_fb_resp:
      resp = np.array((controller.feedback_responses))
      resp_ana = np.array((controller.feedback_responses_ana))

      for i, tit in enumerate(ilc.state_labels):
        for j, ctit in enumerate(ilc.control_labels):

          if np.linalg.norm(resp[:, j, i]) < 1e-4 and (not len(resp_ana) or np.linalg.norm(resp_ana[:, j, i]) < 1e-6):
            continue

          if len(resp_ana):
            if not np.allclose(resp_ana[:, j, i], resp[:, j, i]):
              err = np.mean(np.abs(resp_ana[:, j, i] - resp[:, j, i]))
              print("ERROR: FB resp for", ctit, "/", tit, "doesn't match! Avg. error is", err)

          if args.plot_fb_resp:
            plt.figure()
            plt.plot(ts[:-1], resp[:, j, i], label='d%d / d%d' % (j, i))
            if len(resp_ana):
              plt.plot(ts[:-1], resp_ana[:, j, i], label='d%d / d%d (ana)' % (j, i))

            plt.title("d %s / d %s" % (ctit, tit))
            plt.legend()

    plt.show()

if __name__  == "__main__":
  args = get_parser().parse_args()

  if args.print_params:
    try:
      from tabulate import tabulate

      print(tabulate([
        ["System", args.system],
        ["No. of trials", args.trials],
        ["ILC dt", args.ilc_dt],
        ["Sim dt", args.sim_dt],
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
      print("WARNING: Tabulate import failed; could not print params")

  ILCExperiment(args)
