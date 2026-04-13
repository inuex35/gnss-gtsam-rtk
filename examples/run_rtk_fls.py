#!/usr/bin/env python3
"""RTK-FGO: Factor Graph Optimization RTK with cssrlib + GTSAM.

Usage:
    python run_rtk_fls.py <rover.obs> <base.obs> <nav_file>

Base station position is read from the base RINEX observation header.

Example (Kaiyodai zero-baseline):
    python run_rtk_fls.py ubx.obs rtcm.obs ubx.nav

Environment variables:
    MAX_EP      max epochs (default: all)
    LAG         IFLS lag in seconds, 0=ISAM2 (default: 0)
    SIG_PR      pseudorange sigma [m] (default: 0.3)
    SIG_CP      carrier phase sigma [m] (default: 0.003)
    SIG_DYN     dynamics sigma [m] (default: 0.1)
    SIG_AMB     initial ambiguity sigma [m] (default: 30.0)
    HUBER_PR    Huber threshold for PR, 0=off (default: 0)
    AR_MODE     0:off, 1:continuous, 3:fix-and-hold (default: 3)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.gnss import rSigRnx
from gnss_fgo import GtsamRtk


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_rtk_fls.py <rover.obs> <base.obs> <nav_file>")
        print("  Base position is read from base RINEX header.")
        sys.exit(1)

    obsfile = sys.argv[1]
    basefile = sys.argv[2]
    navfile = sys.argv[3]

    nep = int(os.environ.get('MAX_EP', '9999'))

    # Signal definitions (GPS L1/L2 + QZS L1/L2)
    sigs = [rSigRnx("GC1C"), rSigRnx("GC2X"),
            rSigRnx("GL1C"), rSigRnx("GL2X"),
            rSigRnx("GS1C"), rSigRnx("GS2X"),
            rSigRnx("JC1C"), rSigRnx("JC2X"),
            rSigRnx("JL1C"), rSigRnx("JL2X"),
            rSigRnx("JS1C"), rSigRnx("JS2X")]

    # Decode RINEX
    dec = rn.rnxdec()
    dec.setSignals(sigs)
    nav = gn.Nav()
    dec.decode_nav(navfile, nav)

    decb = rn.rnxdec()
    decb.setSignals(sigs)
    decb.decode_obsh(basefile)
    dec.decode_obsh(obsfile)

    # Base position from RINEX header
    xyz_ref = list(decb.pos)
    if np.linalg.norm(xyz_ref) == 0:
        print("Error: base position not found in RINEX header")
        sys.exit(1)
    pos_ref = gn.ecef2pos(xyz_ref)
    print(f"Base pos: {xyz_ref[0]:.4f} {xyz_ref[1]:.4f} {xyz_ref[2]:.4f}")

    nav.rb = xyz_ref
    nav.pmode = 1
    nav.ephopt = 0

    pos0 = np.array(dec.pos) if np.linalg.norm(dec.pos) > 0 else np.array(xyz_ref)
    rtk = GtsamRtk(nav, pos0)

    lag = float(os.environ.get('LAG', '0'))
    mode = f"IFLS(LAG={lag}s)" if lag > 0 else "ISAM2"
    print(f"RTK-FGO: {mode}, σ_PR={rtk.sigma_pr} σ_CP={rtk.sigma_cp} "
          f"σ_DYN={rtk.sigma_dyn} AR={rtk.nav.armode}")

    enu = []
    smode_list = []
    ne = 0

    for _ in range(nep):
        obs, obsb = rn.sync_obs(dec, decb)
        if obs is None or obsb is None:
            break
        if ne == 0:
            t0 = nav.t = obs.t

        try:
            rtk.process(obs, obsb=obsb)
        except Exception:
            ne += 1
            continue

        sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
        e = gn.ecef2enu(pos_ref, sol - xyz_ref)
        enu.append(e)
        smode_list.append(nav.smode)

        if ne < 10 or ne % 50 == 0:
            tag = "FIX" if nav.smode == 4 else "FLT"
            print(f"Ep {ne:4d}: {tag} E={e[0]:+.4f} N={e[1]:+.4f} "
                  f"U={e[2]:+.4f} 3D={np.linalg.norm(e):.4f}")
        ne += 1

    dec.fobs.close()
    decb.fobs.close()

    # Summary
    enu = np.array(enu)
    smode = np.array(smode_list)
    n = len(enu)

    print(f"\n{'='*50}")
    print(f"Results: {n} epochs")
    fix_mask = smode == 4
    print(f"Fix: {np.sum(fix_mask)}, Float: {np.sum(smode == 5)}")

    if np.any(fix_mask):
        ef = enu[fix_mask]
        print(f"\nFix ({np.sum(fix_mask)} ep):")
        print(f"  E: {np.sqrt(np.mean(ef[:,0]**2)):.4f}m  "
              f"N: {np.sqrt(np.mean(ef[:,1]**2)):.4f}m  "
              f"U: {np.sqrt(np.mean(ef[:,2]**2)):.4f}m  "
              f"3D: {np.sqrt(np.mean(np.sum(ef**2, axis=1))):.4f}m")

    print(f"\nAll ({n} ep):")
    print(f"  E: {np.sqrt(np.mean(enu[:,0]**2)):.4f}m  "
          f"N: {np.sqrt(np.mean(enu[:,1]**2)):.4f}m  "
          f"U: {np.sqrt(np.mean(enu[:,2]**2)):.4f}m  "
          f"3D: {np.sqrt(np.mean(np.sum(enu**2, axis=1))):.4f}m")


if __name__ == "__main__":
    main()
