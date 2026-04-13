#!/usr/bin/env python3
"""RTK positioning using cssrlib observation model + GTSAM optimization.

Replaces cssrlib's kfupdate() with GTSAM factor graph optimization.
The observation model (sdres → H, v, R) comes from cssrlib.
"""

import sys
import os
import numpy as np



import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.rtk import rtkpos
from cssrlib.gnss import rSigRnx, rCST, uGNSS, sat2prn, ecef2pos, timediff
from cssrlib.ephemeris import satposs

import gtsam
from gtsam import symbol

# GTSAM symbol keys
X = lambda i: symbol('x', i)  # position at epoch i
B = lambda i: symbol('b', i)  # bias state index i (maps to nav.x indices)


class GtsamRtkPos(rtkpos):
    """RTK processor using GTSAM instead of Kalman filter."""

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, pos0, logfile)
        self.epoch = 0

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """Process one epoch: cssrlib observation model + GTSAM update."""

        if len(obs.sat) == 0:
            return

        self.nav.nsat[0] = len(obs.sat)

        # Satellite positions
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)
        self.nav.nsat[1] = nsat

        if nsat < 6:
            return

        # Quality control
        sat_ed = self.qcedit(obs, rs, dts, svh)

        # RTK base processing
        y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
        ns = len(iu)
        self.nav.nsat[2] = ns

        if ns < 6:
            return

        # Time propagation (prediction step)
        self.udstate(obs_)

        xp = self.nav.x.copy()
        Pp = self.nav.P.copy()

        # Zero-difference residuals
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])
        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        el = elu[iu]

        self.nav.sat = sat
        self.nav.el[sat-1] = el
        self.nav.y = y

        ny = y.shape[0]
        if ny < 6:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 5
            return -1

        # SD residuals → H, v, R
        v, H, R = self.sdres(obs, xp, y, e, sat, el)

        # ============================================================
        # GTSAM update instead of kfupdate
        # ============================================================
        xp, Pp = self.gtsam_update(xp, Pp, H, v, R)

        # Post-fit residuals
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])
        y_post = yu[iu, :]
        e_post = eu[iu, :]
        if y_post.shape[0] < 6:
            return -1

        v_post, H_post, R_post = self.sdres(obs, xp, y_post, e_post, sat, el)
        if self.valpos(v_post, R_post):
            self.nav.x = xp
            self.nav.P = Pp
            self.nav.ns = 0
            for i in range(ns):
                j = sat[i]-1
                for f in range(self.nav.nf):
                    if self.nav.vsat[j, f] == 0:
                        continue
                    self.nav.outc[j, f] = 0
                    if f == 0:
                        self.nav.ns += 1
        else:
            self.nav.smode = 0

        self.nav.smode = 5  # float

        # Ambiguity resolution
        if self.nav.armode > 0:
            nb, xa = self.resamb_lambda(sat, self.nav.parmode, self.nav.par_P0)
            if nb > 0:
                yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xa[0:3])
                y_fix = yu[iu, :]
                e_fix = eu[iu, :]
                v_fix, H_fix, R_fix = self.sdres(obs, xa, y_fix, e_fix, sat, el)
                if self.valpos(v_fix, R_fix):
                    if self.nav.armode == 3:
                        self.holdamb(xa)
                    self.nav.smode = 4  # fix

        self.nav.t = obs.t
        self.epoch += 1
        return 0

    def gtsam_update(self, x, P, H, v, R):
        """Replace Kalman filter update with GTSAM Gaussian factor graph.

        Compress to active states (non-zero diagonal in P) to avoid
        singular information matrix, same as RTKLIB's filter_().
        """
        nx = len(x)
        nv = len(v)

        # Find active states (initialized, non-zero variance)
        dP = np.diag(P)
        active = np.where((x != 0) | (dP > 0))[0]
        # Must include position states 0:3 always
        active = np.union1d(np.arange(min(3, nx)), active).astype(int)

        # Also include states that H references
        h_active = np.where(np.any(H != 0, axis=0))[0]
        active = np.union1d(active, h_active).astype(int)

        na = len(active)
        if na == 0:
            return x, P

        x_a = x[active]
        P_a = P[np.ix_(active, active)]
        H_a = H[:, active]

        # Regularize P_a: ensure positive definite
        dPa = np.diag(P_a).copy()
        for i in range(na):
            if dPa[i] <= 0:
                dPa[i] = 1e6  # large variance for uninitialized
        P_a[np.diag_indices(na)] = np.maximum(np.diag(P_a), 1e-10)

        # Build GTSAM Gaussian factor graph
        graph = gtsam.GaussianFactorGraph()
        key = 0

        # 1. Prior: information form from predicted P
        try:
            P_a_inv = np.linalg.inv(P_a)
        except np.linalg.LinAlgError:
            P_a_inv = np.linalg.pinv(P_a)
        P_a_inv = 0.5 * (P_a_inv + P_a_inv.T)

        graph.add(gtsam.HessianFactor(key, P_a_inv, np.zeros(na), 0.0))

        # 2. Measurement: v = H_a @ dx + noise(R)
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
        R_inv = 0.5 * (R_inv + R_inv.T)

        HtRinv = H_a.T @ R_inv
        G = HtRinv @ H_a
        g = HtRinv @ v
        graph.add(gtsam.HessianFactor(key, G, g, 0.0))

        # 3. Solve for dx
        try:
            result = graph.optimize()
            dx = result.at(key)
        except RuntimeError:
            # Fallback to numpy KF
            PHt = P_a @ H_a.T
            S = H_a.T @ PHt + R
            try:
                K = PHt @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return x, P
            dx = K @ v

        x_a_new = x_a + dx

        # 4. Posterior covariance
        Lambda_post = P_a_inv + G
        try:
            P_a_new = np.linalg.inv(Lambda_post)
        except np.linalg.LinAlgError:
            P_a_new = np.linalg.pinv(Lambda_post)
        P_a_new = 0.5 * (P_a_new + P_a_new.T)

        # Write back to full state
        x_new = x.copy()
        P_new = P.copy()
        x_new[active] = x_a_new
        for i, gi in enumerate(active):
            for j, gj in enumerate(active):
                P_new[gi, gj] = P_a_new[i, j]

        return x_new, P_new


# ============================================================
# Main
# ============================================================

if len(sys.argv) < 4:
    print("Usage: python run_rtk_fls.py <rover.obs> <base.obs> <nav_file>")
    sys.exit(1)

obsfile = sys.argv[1]
basefile = sys.argv[2]
navfile = sys.argv[3]

sigs = [rSigRnx("GC1C"), rSigRnx("GC2X"),
        rSigRnx("GL1C"), rSigRnx("GL2X"), rSigRnx("GS1C"), rSigRnx("GS2X"),
        rSigRnx("JC1C"), rSigRnx("JC2X"),
        rSigRnx("JL1C"), rSigRnx("JL2X"), rSigRnx("JS1C"), rSigRnx("JS2X")]

dec = rn.rnxdec()
dec.setSignals(sigs)
nav = gn.Nav()
dec.decode_nav(navfile, nav)

decb = rn.rnxdec()
decb.setSignals(sigs)
decb.decode_obsh(basefile)
dec.decode_obsh(obsfile)

xyz_ref = list(decb.pos)
if np.linalg.norm(xyz_ref) == 0:
    print("Error: base position not found in RINEX header")
    sys.exit(1)
pos_ref = gn.ecef2pos(xyz_ref)
print(f"Base pos: {xyz_ref[0]:.4f} {xyz_ref[1]:.4f} {xyz_ref[2]:.4f}")

nep = int(os.environ.get('MAX_EP', '9999'))
nav.rb = xyz_ref
nav.pmode = 1
nav.ephopt = 0

pos0 = np.array(dec.pos) if np.linalg.norm(dec.pos) > 0 else np.array(xyz_ref)
rtk = GtsamRtkPos(nav, pos0, 'rtk_fls.log')

t = np.zeros(nep)
enu = np.zeros((nep, 3))
smode = np.zeros(nep, dtype=int)
err3d = np.zeros(nep)

print(f"cssrlib observation model + GTSAM optimization")
print(f"Rover pos0: {pos0}")
print(f"Base  pos:  {xyz_ref}")
print(f"Epochs: {nep}")
print()

ne_actual = 0
for ne in range(nep):
    obs, obsb = rn.sync_obs(dec, decb)
    if obs is None or obsb is None:
        break
    if ne == 0:
        t0 = nav.t = obs.t

    try:
        rtk.process(obs, obsb=obsb)
    except Exception as e:
        if ne < 5:
            print(f"Ep {ne}: error: {e}")
        continue

    t[ne] = gn.timediff(nav.t, t0)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    enu[ne, :] = gn.ecef2enu(pos_ref, sol - xyz_ref)
    smode[ne] = nav.smode
    err3d[ne] = np.linalg.norm(enu[ne, :])
    ne_actual = ne + 1

    if ne < 10 or ne % 50 == 0:
        tag = "FIX" if nav.smode == 4 else "FLT"
        print(f"Ep {ne:3d}: {tag} E={enu[ne,0]:+.4f} N={enu[ne,1]:+.4f} "
              f"U={enu[ne,2]:+.4f} 3D={err3d[ne]:.4f}")

dec.fobs.close()
decb.fobs.close()

# Summary
print(f"\n{'='*60}")
print(f"cssrlib + GTSAM RTK Results: {ne_actual} epochs")
print(f"{'='*60}")

enu = enu[:ne_actual]
smode = smode[:ne_actual]
err3d = err3d[:ne_actual]

fix_mask = smode == 4
flt_mask = smode == 5
print(f"Fix: {np.sum(fix_mask)}, Float: {np.sum(flt_mask)}, "
      f"None: {np.sum((smode != 4) & (smode != 5))}")

if np.any(fix_mask):
    ef = enu[fix_mask]
    e3f = np.linalg.norm(ef, axis=1)
    print(f"\nFix ({np.sum(fix_mask)} epochs):")
    print(f"  3D RMS: {np.sqrt(np.mean(e3f**2)):.4f}m")
    print(f"  E RMS:  {np.sqrt(np.mean(ef[:,0]**2)):.4f}m")
    print(f"  N RMS:  {np.sqrt(np.mean(ef[:,1]**2)):.4f}m")
    print(f"  U RMS:  {np.sqrt(np.mean(ef[:,2]**2)):.4f}m")

if np.any(flt_mask):
    ef = enu[flt_mask]
    print(f"\nFloat ({np.sum(flt_mask)} epochs):")
    print(f"  3D RMS: {np.sqrt(np.mean(np.linalg.norm(ef,axis=1)**2)):.4f}m")

print(f"\nAll ({ne_actual} epochs):")
print(f"  3D RMS: {np.sqrt(np.mean(err3d**2)):.4f}m, median: {np.median(err3d):.4f}m")
print(f"  E RMS:  {np.sqrt(np.mean(enu[:,0]**2)):.4f}m")
print(f"  N RMS:  {np.sqrt(np.mean(enu[:,1]**2)):.4f}m")
print(f"  U RMS:  {np.sqrt(np.mean(enu[:,2]**2)):.4f}m")
