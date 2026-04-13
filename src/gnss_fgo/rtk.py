"""GTSAM RTK: Factor graph RTK using cssrlib obs model + GTSAM ISAM2.

No EKF — position from GTSAM, covariance from GTSAM Marginals.
DD factors (C++ nonlinear), AR via cssrlib LAMBDA, fix-and-hold.
"""

import os
import numpy as np
import gtsam
from cssrlib.rtk import rtkpos
from cssrlib.gnss import uGNSS, uTYP, rCST, sat2prn, geodist, timediff
from cssrlib.ephemeris import satposs


class GtsamRtk(rtkpos):

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, pos0, logfile)
        self.epoch = 0
        self.epoch_time = 0.0

        lag = float(os.environ.get('LAG', '0'))
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        if lag > 0:
            self.smoother = gtsam.IncrementalFixedLagSmoother(lag, params)
            self.isam = None
        else:
            self.isam = gtsam.ISAM2(params)
            self.smoother = None

        self.amb_keys = {}
        self.current_estimate = None
        self.sigma_pr = float(os.environ.get('SIG_PR', '0.3'))
        self.sigma_cp = float(os.environ.get('SIG_CP', '0.003'))
        self.sigma_dyn = float(os.environ.get('SIG_DYN', '0.1'))
        self.sigma_amb0 = float(os.environ.get('SIG_AMB', '30.0'))
        self.huber_pr = float(os.environ.get('HUBER_PR', '0'))
        self.nav.armode = int(os.environ.get('AR_MODE', '3'))

    def X(self, ep): return gtsam.symbol('x', ep)
    def N(self, sat, f): return gtsam.symbol('n', sat * 10 + f)

    def _get_wavelengths(self, obs, sat):
        sys_i, _ = sat2prn(sat)
        if sys_i not in obs.sig:
            return []
        sigs = obs.sig[sys_i][uTYP.L]
        if sys_i == uGNSS.GLO:
            return [s.wavelength(self.nav.glo_ch.get(sat, 0)) for s in sigs]
        return [s.wavelength() for s in sigs]

    def _manage_ambiguities(self, obs):
        """Cycle slip detection + new ambiguity initialization."""
        ns = len(obs.sat)
        sat = obs.sat
        for f in range(self.nav.nf):
            for i in range(uGNSS.MAXSAT):
                self.nav.outc[i, f] += 1
                sat_ = i + 1
                sys_i, _ = sat2prn(sat_)
                reset = (self.nav.outc[i, f] > self.nav.maxout
                         or np.any(self.nav.edt[i, :] > 0))
                if sys_i not in obs.sig.keys():
                    continue
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0

            for i in range(ns):
                if np.any(self.nav.edt[sat[i]-1, :] > 0):
                    continue
                sys_i, _ = sat2prn(sat[i])
                if sys_i not in obs.sig.keys():
                    continue
                sig = obs.sig[sys_i][uTYP.L][f]
                fi = (sig.frequency(self.nav.glo_ch.get(sat[i], 0))
                      if sys_i == uGNSS.GLO else sig.frequency())
                lam = rCST.CLIGHT / fi if fi > 0 else 0
                cp, pr = obs.L[i, f], obs.P[i, f]
                if cp == 0 or pr == 0 or lam == 0:
                    continue
                j = self.IB(sat[i], f, self.nav.na)
                if self.nav.x[j] == 0.0:
                    self.initx(cp - pr / lam, self.nav.sig_n0**2, j)

    def _build_dd_factors(self, graph, new_values, obs, obsb, obs_sd,
                          rs, rsb, sat, el, iu, ir, pos_pred, ep):
        """Build DD-PR and DD-CP factors, initialize new ambiguities."""
        nv = 0
        new_amb = {}
        base_pt = gtsam.Point3(*self.nav.rb)
        rb = np.array(self.nav.rb)
        key_x = self.X(ep)

        for sys_id in obs_sd.sig.keys():
            idx_sys = self.sysidx(sat, sys_id)
            if len(idx_sys) < 2:
                continue
            ref_idx = idx_sys[np.argmax(el[idx_sys])]
            ref_sat = sat[ref_idx]
            lams = self._get_wavelengths(obs_sd, ref_sat)

            for j_idx in idx_sys:
                if j_idx == ref_idx:
                    continue
                sat_j = sat[j_idx]
                ri, ji = ref_idx, j_idx

                for f in range(self.nav.nf):
                    if f >= len(lams):
                        continue
                    lam_f = lams[f]
                    if obs_sd.P[ri, f] == 0 or obs_sd.P[ji, f] == 0:
                        continue

                    rs_ref = gtsam.Point3(*rs[iu[ri], :3])
                    rs_j = gtsam.Point3(*rs[iu[ji], :3])
                    rsb_ref = gtsam.Point3(*rsb[ir[ri], :3])
                    rsb_j = gtsam.Point3(*rsb[ir[ji], :3])

                    has_cp = obs_sd.L[ri, f] != 0 and obs_sd.L[ji, f] != 0
                    if has_cp:
                        kr = self.N(ref_sat, f)
                        kj = self.N(sat_j, f)
                        for sn, kn, sd_cp, rs_s in [
                                (ref_sat, kr, obs_sd.L[ri, f]*lam_f, rs[iu[ri], :3]),
                                (sat_j, kj, obs_sd.L[ji, f]*lam_f, rs[iu[ji], :3])]:
                            if (sn, f) not in self.amb_keys and (sn, f) not in new_amb:
                                r_r, _ = geodist(rs_s, pos_pred)
                                r_b, _ = geodist(rs_s, rb)
                                n0 = sd_cp - (r_r - r_b)
                                new_values.insert(kn, n0)
                                graph.addPriorDouble(kn, n0,
                                    gtsam.noiseModel.Isotropic.Sigma(1, self.sigma_amb0))
                                new_amb[(sn, f)] = kn
                                self.nav.x[self.IB(sn, f, self.nav.na)] = n0

                        graph.add(gtsam.DDCarrierPhaseFactor(
                            key_x, kr, kj,
                            float(obs.L[iu[ri], f]) * lam_f,
                            float(obsb.L[ir[ri], f]) * lam_f,
                            float(obs.L[iu[ji], f]) * lam_f,
                            float(obsb.L[ir[ji], f]) * lam_f,
                            rs_ref, rs_j, rsb_ref, rsb_j, base_pt, lam_f,
                            gtsam.noiseModel.Isotropic.Sigma(1, self.sigma_cp * np.sqrt(2))))
                        nv += 1
                    else:
                        noise_pr = gtsam.noiseModel.Isotropic.Sigma(
                            1, self.sigma_pr * np.sqrt(2))
                        if self.huber_pr > 0:
                            noise_pr = gtsam.noiseModel.Robust.Create(
                                gtsam.noiseModel.mEstimator.Huber.Create(self.huber_pr),
                                noise_pr)
                        graph.add(gtsam.DDPseudorangeFactor(
                            key_x,
                            float(obs.P[iu[ri], f]), float(obsb.P[ir[ri], f]),
                            float(obs.P[iu[ji], f]), float(obsb.P[ir[ji], f]),
                            rs_ref, rs_j, rsb_ref, rsb_j, base_pt, noise_pr))
                        nv += 1

        self.amb_keys.update(new_amb)
        return nv, new_amb

    def _write_back(self, estimate, key_x):
        """Write GTSAM estimate + Marginals to nav.x/P."""
        self.nav.P[:, :] = 0
        self.nav.vsat[:, :] = 0
        self.nav.x[0:3] = np.array(estimate.atPoint3(key_x))

        for (s, f), k in self.amb_keys.items():
            if estimate.exists(k):
                self.nav.x[self.IB(s, f, self.nav.na)] = estimate.atDouble(k)
                self.nav.vsat[s - 1, f] = 1

        try:
            mg = (gtsam.Marginals(self.smoother.getFactors(), estimate)
                  if self.smoother else
                  gtsam.Marginals(self.isam.getFactorsUnsafe(), estimate))
            active = [(s, f, k) for (s, f), k in self.amb_keys.items()
                      if estimate.exists(k)]
            if active:
                keys = gtsam.KeyVector()
                keys.append(key_x)
                for s, f, k in active:
                    keys.append(k)
                jm = mg.jointMarginalCovariance(keys)
                self.nav.P[0:3, 0:3] = jm.at(key_x, key_x)
                for s, f, k in active:
                    idx = self.IB(s, f, self.nav.na)
                    Pxn = jm.at(key_x, k)
                    self.nav.P[0:3, idx] = Pxn[:, 0]
                    self.nav.P[idx, 0:3] = Pxn[:, 0]
                    self.nav.P[idx, idx] = jm.at(k, k)[0, 0]
                for i, (s1, f1, k1) in enumerate(active):
                    i1 = self.IB(s1, f1, self.nav.na)
                    for j, (s2, f2, k2) in enumerate(active):
                        if i >= j:
                            continue
                        i2 = self.IB(s2, f2, self.nav.na)
                        c = jm.at(k1, k2)[0, 0]
                        self.nav.P[i1, i2] = c
                        self.nav.P[i2, i1] = c
            else:
                self.nav.P[0:3, 0:3] = mg.marginalCovariance(key_x)
        except Exception:
            pass

    def _do_ar(self, obs, rs, vs, dts, sat, el, iu):
        """LAMBDA AR + fix-and-hold."""
        try:
            nb, xa = self.resamb_lambda(sat, self.nav.parmode, self.nav.par_P0)
        except (SystemExit, Exception):
            return
        if nb <= 0:
            return
        yu, eu, _ = self.zdres(obs, None, None, rs, vs, dts, xa[0:3])
        v_fix, _, R_fix = self.sdres(obs, xa, yu[iu], eu[iu], sat, el)
        if not self.valpos(v_fix, R_fix):
            return
        if self.nav.armode == 3:
            self.holdamb(xa)
            self._inject_hold(xa)
        self.nav.smode = 4

    def _inject_hold(self, xa):
        """Inject held ambiguities as tight priors into ISAM2."""
        graph = gtsam.NonlinearFactorGraph()
        for (s, f), k in self.amb_keys.items():
            if self.nav.fix[s - 1, f] == 3:
                graph.addPriorDouble(k, xa[self.IB(s, f, self.nav.na)],
                    gtsam.noiseModel.Isotropic.Sigma(1, np.sqrt(self.VAR_HOLDAMB)))
        if graph.size() > 0:
            try:
                if self.smoother:
                    self.smoother.update(graph, gtsam.Values(),
                                         gtsam.FixedLagSmootherKeyTimestampMap())
                elif self.isam:
                    self.isam.update(graph, gtsam.Values())
            except RuntimeError:
                pass

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        if len(obs.sat) == 0:
            return

        # Satellite positions + quality control
        self.nav.nsat[0] = len(obs.sat)
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        self.nav.nsat[1] = nsat
        if nsat < 6:
            return
        self.qcedit(obs, rs, dts, svh)

        # SD observations
        y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
        ns = len(iu)
        self.nav.nsat[2] = ns
        if ns < 6:
            return

        # Base satellite positions
        sat_common = obs.sat[iu]
        ir = np.array([np.where(obsb.sat == s)[0][0]
                        for s in sat_common if s in obsb.sat])
        rsb, _, _, _, _ = satposs(obsb, self.nav)

        # Ambiguity management (no EKF)
        self._manage_ambiguities(obs_)
        sat = obs.sat[iu]

        # Position from GTSAM previous estimate
        if self.current_estimate is not None and self.epoch > 0:
            prev = self.X(self.epoch - 1)
            pos_pred = (np.array(self.current_estimate.atPoint3(prev))
                        if self.current_estimate.exists(prev)
                        else self.nav.x[0:3].copy())
        else:
            pos_pred = self.nav.x[0:3].copy()

        # Elevation
        yu, eu, elu = self.zdres(obs, None, None, rs, vs, dts, pos_pred)
        el = elu[iu]
        self.nav.sat = sat
        self.nav.el[sat - 1] = el
        if ns < 6:
            return

        # Factor graph
        graph = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()
        ep = self.epoch
        key_x = self.X(ep)

        if ep == 0:
            graph.addPriorPoint3(key_x, gtsam.Point3(*pos_pred),
                                 gtsam.noiseModel.Isotropic.Sigma(3, self.nav.sig_p0))
        else:
            dt = timediff(obs.t, self.nav.t) if self.nav.t.time > 0 else 1.0
            graph.add(gtsam.BetweenFactorPoint3(
                self.X(ep - 1), key_x, gtsam.Point3(0, 0, 0),
                gtsam.noiseModel.Isotropic.Sigma(
                    3, self.sigma_dyn * np.sqrt(max(abs(dt), 0.1)))))
        new_values.insert(key_x, gtsam.Point3(*pos_pred))

        nv, new_amb = self._build_dd_factors(
            graph, new_values, obs, obsb, obs_, rs, rsb, sat, el, iu, ir, pos_pred, ep)
        if nv < 4:
            self.epoch += 1
            self.nav.t = obs.t
            return

        # ISAM2 / IFLS update
        try:
            if self.smoother:
                ts = gtsam.FixedLagSmootherKeyTimestampMap()
                ts[key_x] = self.epoch_time
                for (s, f), k in new_amb.items():
                    ts[k] = self.epoch_time
                for (s, f), k in self.amb_keys.items():
                    if (s, f) not in new_amb:
                        ts[k] = self.epoch_time
                self.smoother.update(graph, new_values, ts)
                estimate = self.smoother.calculateEstimate()
            else:
                self.isam.update(graph, new_values)
                self.isam.update()
                estimate = self.isam.calculateEstimate()
            self.current_estimate = estimate
        except RuntimeError:
            self.epoch += 1
            self.epoch_time += 1.0
            self.nav.t = obs.t
            return

        self._write_back(estimate, key_x)
        self.nav.smode = 5

        if self.nav.armode > 0:
            self._do_ar(obs, rs, vs, dts, sat, el, iu)

        self.nav.t = obs.t
        self.epoch += 1
        self.epoch_time += 1.0
