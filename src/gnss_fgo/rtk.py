"""GTSAM-based RTK processor using cssrlib observation model.

Replaces cssrlib's EKF (sdres + kfupdate) with GTSAM ISAM2 / IFLS.
DD-PR and DD-CP as C++ nonlinear factors (DDPseudorangeFactor, DDCarrierPhaseFactor).
Ambiguity resolution via cssrlib's LAMBDA.
"""

import os
import numpy as np
import gtsam

from cssrlib.rtk import rtkpos
from cssrlib.gnss import uGNSS, uTYP, sat2prn, geodist, timediff
from cssrlib.ephemeris import satposs


class GtsamRtk(rtkpos):
    """RTK processor: cssrlib obs model + GTSAM factor graph optimizer."""

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, pos0, logfile)
        self.epoch = 0
        self.epoch_time = 0.0

        # Optimizer
        self.smoother_lag = float(os.environ.get('LAG', '0'))
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1

        if self.smoother_lag > 0:
            self.smoother = gtsam.IncrementalFixedLagSmoother(
                self.smoother_lag, params)
            self.isam = None
        else:
            self.isam = gtsam.ISAM2(params)
            self.smoother = None

        self.amb_keys = {}
        self.current_estimate = None

        # Noise
        self.sigma_pr = float(os.environ.get('SIG_PR', '0.3'))
        self.sigma_cp = float(os.environ.get('SIG_CP', '0.003'))
        self.sigma_dyn = float(os.environ.get('SIG_DYN', '0.1'))
        self.sigma_amb0 = float(os.environ.get('SIG_AMB', '30.0'))
        self.huber_pr = float(os.environ.get('HUBER_PR', '0'))

        ar = int(os.environ.get('AR_MODE', '3'))
        self.nav.armode = ar

    # -- Key helpers --
    def X(self, ep):
        return gtsam.symbol('x', ep)

    def N(self, sat, f):
        return gtsam.symbol('n', sat * 10 + f)

    # -- Observation helpers --
    def get_wavelengths(self, obs, sat):
        sys_i, _ = sat2prn(sat)
        if sys_i not in obs.sig:
            return []
        sigsCP = obs.sig[sys_i][uTYP.L]
        if sys_i == uGNSS.GLO:
            return [s.wavelength(self.nav.glo_ch.get(sat, 0)) for s in sigsCP]
        return [s.wavelength() for s in sigsCP]

    def build_dd_pairs(self, obs_sd, rs, sat, el, iu):
        """Build DD observation pairs for C++ factors.

        Returns list of dicts with ref/target sat info, raw obs indices.
        """
        nf = self.nav.nf
        pairs = []

        for sys_id in obs_sd.sig.keys():
            idx_sys = self.sysidx(sat, sys_id)
            if len(idx_sys) < 2:
                continue

            ref_idx = idx_sys[np.argmax(el[idx_sys])]
            ref_sat = sat[ref_idx]
            lams = self.get_wavelengths(obs_sd, ref_sat)

            for j_idx in idx_sys:
                if j_idx == ref_idx:
                    continue
                j_sat = sat[j_idx]

                for f in range(nf):
                    if f >= len(lams):
                        continue
                    lam_f = lams[f]

                    sd_pr_ref = obs_sd.P[ref_idx, f]
                    sd_pr_j = obs_sd.P[j_idx, f]
                    if sd_pr_ref == 0 or sd_pr_j == 0:
                        continue

                    sd_cp_ref = obs_sd.L[ref_idx, f] * lam_f
                    sd_cp_j = obs_sd.L[j_idx, f] * lam_f

                    pairs.append({
                        'ref_sat': ref_sat, 'j_sat': j_sat,
                        'ref_idx': ref_idx, 'j_idx': j_idx,
                        'rs_ref': rs[iu[ref_idx], :3].copy(),
                        'rs_j': rs[iu[j_idx], :3].copy(),
                        'sd_pr_ref': sd_pr_ref, 'sd_pr_j': sd_pr_j,
                        'sd_cp_ref': sd_cp_ref, 'sd_cp_j': sd_cp_j,
                        'has_cp': sd_cp_ref != 0 and sd_cp_j != 0,
                        'freq': f, 'lam': lam_f,
                    })

        return pairs

    # -- Main processing --
    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        if len(obs.sat) == 0:
            return

        self.nav.nsat[0] = len(obs.sat)
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        self.nav.nsat[1] = nsat
        if nsat < 6:
            return

        self.qcedit(obs, rs, dts, svh)

        # Base processing (SD observations)
        y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
        ns = len(iu)
        self.nav.nsat[2] = ns
        if ns < 6:
            return

        # Raw rover/base obs for C++ factor API
        sat_common = obs.sat[iu]
        ir = np.array([np.where(obsb.sat == s)[0][0]
                        for s in sat_common if s in obsb.sat])
        rsb, _, _, _, _ = satposs(obsb, self.nav)

        # Time propagation
        self.udstate(obs_)
        xp = self.nav.x.copy()
        pos_pred = xp[0:3].copy()
        sat = obs.sat[iu]

        yu, eu, elu = self.zdres(obs, None, None, rs, vs, dts, pos_pred)
        el = elu[iu]
        self.nav.sat = sat
        self.nav.el[sat - 1] = el

        if ns < 6:
            return

        # DD pairs
        dd_pairs = self.build_dd_pairs(obs_, rs, sat, el, iu)

        # ---- Factor Graph ----
        graph = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()
        ep = self.epoch
        key_x = self.X(ep)

        # Position prior / dynamics
        if ep == 0:
            graph.addPriorPoint3(key_x, gtsam.Point3(*pos_pred),
                                 gtsam.noiseModel.Isotropic.Sigma(3, self.nav.sig_p0))
        else:
            dt = timediff(obs.t, self.nav.t) if self.nav.t.time > 0 else 1.0
            sig_dyn = self.sigma_dyn * np.sqrt(max(abs(dt), 0.1))
            graph.add(gtsam.BetweenFactorPoint3(
                self.X(ep - 1), key_x, gtsam.Point3(0, 0, 0),
                gtsam.noiseModel.Isotropic.Sigma(3, sig_dyn)))

        new_values.insert(key_x, gtsam.Point3(*pos_pred))

        # DD factors
        nv = 0
        new_amb_keys = {}
        base_pt = gtsam.Point3(*self.nav.rb)
        rb = np.array(self.nav.rb)

        for dd in dd_pairs:
            ref_sat, sat_j = dd['ref_sat'], dd['j_sat']
            f, lam_f = dd['freq'], dd['lam']
            ri = dd['ref_idx']
            ji = dd['j_idx']

            rs_ref_pt = gtsam.Point3(*dd['rs_ref'])
            rs_j_pt = gtsam.Point3(*dd['rs_j'])
            rs_ref_base_pt = gtsam.Point3(*rsb[ir[ri], :3])
            rs_j_base_pt = gtsam.Point3(*rsb[ir[ji], :3])

            if dd['has_cp']:
                key_n_ref = self.N(ref_sat, f)
                key_n_j = self.N(sat_j, f)

                # Initialize ambiguities
                for sat_n, key_n, sd_cp, rs_sat in [
                        (ref_sat, key_n_ref, dd['sd_cp_ref'], dd['rs_ref']),
                        (sat_j, key_n_j, dd['sd_cp_j'], dd['rs_j'])]:
                    if (sat_n, f) not in self.amb_keys and \
                       (sat_n, f) not in new_amb_keys:
                        r_rov, _ = geodist(rs_sat, pos_pred)
                        r_base, _ = geodist(rs_sat, rb)
                        n0 = sd_cp - (r_rov - r_base)
                        new_values.insert(key_n, n0)
                        graph.addPriorDouble(key_n, n0,
                            gtsam.noiseModel.Isotropic.Sigma(1, self.sigma_amb0))
                        new_amb_keys[(sat_n, f)] = key_n
                        self.nav.x[self.IB(sat_n, f, self.nav.na)] = n0

                # DD-CP factor (raw rover/base CP)
                cp_rov_ref = float(obs.L[iu[ri], f]) * lam_f
                cp_base_ref = float(obsb.L[ir[ri], f]) * lam_f
                cp_rov_j = float(obs.L[iu[ji], f]) * lam_f
                cp_base_j = float(obsb.L[ir[ji], f]) * lam_f

                noise_cp = gtsam.noiseModel.Isotropic.Sigma(
                    1, self.sigma_cp * np.sqrt(2))
                graph.add(gtsam.DDCarrierPhaseFactor(
                    key_x, key_n_ref, key_n_j,
                    cp_rov_ref, cp_base_ref, cp_rov_j, cp_base_j,
                    rs_ref_pt, rs_j_pt, rs_ref_base_pt, rs_j_base_pt,
                    base_pt, lam_f, noise_cp))
                nv += 1

            else:
                # DD-PR factor (raw rover/base PR)
                pr_rov_ref = float(obs.P[iu[ri], f])
                pr_base_ref = float(obsb.P[ir[ri], f])
                pr_rov_j = float(obs.P[iu[ji], f])
                pr_base_j = float(obsb.P[ir[ji], f])

                noise_pr = gtsam.noiseModel.Isotropic.Sigma(
                    1, self.sigma_pr * np.sqrt(2))
                if self.huber_pr > 0:
                    noise_pr = gtsam.noiseModel.Robust.Create(
                        gtsam.noiseModel.mEstimator.Huber.Create(self.huber_pr),
                        noise_pr)
                graph.add(gtsam.DDPseudorangeFactor(
                    key_x,
                    pr_rov_ref, pr_base_ref, pr_rov_j, pr_base_j,
                    rs_ref_pt, rs_j_pt, rs_ref_base_pt, rs_j_base_pt,
                    base_pt, noise_pr))
                nv += 1

        if nv < 4:
            self.epoch += 1
            self.nav.t = obs.t
            return

        self.amb_keys.update(new_amb_keys)

        # ---- Optimizer update ----
        try:
            if self.smoother is not None:
                timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
                timestamps[key_x] = self.epoch_time
                for (sat_n, f), key_n in new_amb_keys.items():
                    timestamps[key_n] = self.epoch_time
                for (sat_n, f), key_n in self.amb_keys.items():
                    if (sat_n, f) not in new_amb_keys:
                        timestamps[key_n] = self.epoch_time
                self.smoother.update(graph, new_values, timestamps)
                estimate = self.smoother.calculateEstimate()
            else:
                self.isam.update(graph, new_values)
                self.isam.update()
                estimate = self.isam.calculateEstimate()
            self.current_estimate = estimate
        except RuntimeError as ex:
            self.epoch += 1
            self.epoch_time += 1.0
            self.nav.t = obs.t
            return

        # ---- Write back to nav ----
        self.nav.x[0:3] = np.array(estimate.atPoint3(key_x))
        for (sat_n, f), key_n in self.amb_keys.items():
            if estimate.exists(key_n):
                self.nav.x[self.IB(sat_n, f, self.nav.na)] = estimate.atDouble(key_n)
                self.nav.vsat[sat_n - 1, f] = 1

        # ---- Covariance for LAMBDA ----
        try:
            if self.smoother is not None:
                marginals = gtsam.Marginals(self.smoother.getFactors(), estimate)
            else:
                marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), estimate)

            active_ambs = [(s, f, k) for (s, f), k in self.amb_keys.items()
                           if estimate.exists(k)]
            if active_ambs:
                all_keys = gtsam.KeyVector()
                all_keys.append(key_x)
                for s, f, k in active_ambs:
                    all_keys.append(k)
                jm = marginals.jointMarginalCovariance(all_keys)
                self.nav.P[0:3, 0:3] = jm.at(key_x, key_x)
                for s, f, k in active_ambs:
                    idx = self.IB(s, f, self.nav.na)
                    P_xn = jm.at(key_x, k)
                    self.nav.P[0:3, idx] = P_xn[:, 0]
                    self.nav.P[idx, 0:3] = P_xn[:, 0]
                    self.nav.P[idx, idx] = jm.at(k, k)[0, 0]
                for i, (s1, f1, k1) in enumerate(active_ambs):
                    idx1 = self.IB(s1, f1, self.nav.na)
                    for j, (s2, f2, k2) in enumerate(active_ambs):
                        if i >= j:
                            continue
                        idx2 = self.IB(s2, f2, self.nav.na)
                        cov = jm.at(k1, k2)[0, 0]
                        self.nav.P[idx1, idx2] = cov
                        self.nav.P[idx2, idx1] = cov
            else:
                self.nav.P[0:3, 0:3] = marginals.marginalCovariance(key_x)
        except Exception:
            pass

        self.nav.smode = 5  # float

        # ---- LAMBDA AR (cssrlib resamb_lambda, same as test_cssrlib_gtsam_nl.py) ----
        if self.nav.armode > 0:
            try:
                nb, xa = self.resamb_lambda(
                    sat, self.nav.parmode, self.nav.par_P0)
            except (SystemExit, Exception):
                nb = 0
                xa = None
            if nb > 0:
                yu2, eu2, _ = self.zdres(obs, None, None, rs, vs, dts, xa[0:3])
                v_fix, _, R_fix = self.sdres(obs, xa, yu2[iu], eu2[iu], sat, el)
                if self.valpos(v_fix, R_fix):
                    if self.nav.armode == 3:
                        self.holdamb(xa)
                        self._inject_hold_isam(xa)
                    self.nav.smode = 4

        self.nav.t = obs.t
        self.epoch += 1
        self.epoch_time += 1.0

    def _inject_hold_isam(self, xa):
        """Inject integer ambiguity hold as tight priors into ISAM2/IFLS.

        Same as test_cssrlib_gtsam_nl.py _inject_hold_isam.
        """
        graph = gtsam.NonlinearFactorGraph()
        for (sat_n, f), key_n in self.amb_keys.items():
            if self.nav.fix[sat_n - 1, f] == 3:  # held
                idx = self.IB(sat_n, f, self.nav.na)
                hold_val = xa[idx]
                graph.addPriorDouble(key_n, hold_val,
                    gtsam.noiseModel.Isotropic.Sigma(1, np.sqrt(self.VAR_HOLDAMB)))
        if graph.size() > 0:
            try:
                if self.smoother is not None:
                    self.smoother.update(graph, gtsam.Values(),
                                         gtsam.FixedLagSmootherKeyTimestampMap())
                elif self.isam is not None:
                    self.isam.update(graph, gtsam.Values())
            except RuntimeError:
                pass
