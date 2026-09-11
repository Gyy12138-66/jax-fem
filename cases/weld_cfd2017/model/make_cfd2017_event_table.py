#!/usr/bin/env python3
"""Event table for the 2017 hybrid laser-MIG CFD case (Desktop/熔池温度场result, input_param.txt).

Process (full-plate frame, weld along +y at x = x_centre, top surface z = thickness):
  laser/arc on from t = 0 to 1.5 s, source starts at y = 8 mm and travels at 18 mm/s (to y = 35 mm),
  then off; the CFD stops at 1.7 s, this table continues cooling to --cool-to seconds.
Single EQUIVALENT source (deviation D-C-01): absorbed power = arc 0.7*3200 + laser 0.3*1000 = 2540 W
(full plate; the CFD log's 1118 + 150 W are half-domain values). Run with --absorptivity 1.0.
Rows every --dt during heating (CFD used 5 ms); cooling rows grow geometrically.
"""
import argparse
import csv
import json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="inputs/cfd2017_single_source_path.csv")
    p.add_argument("--report", default="inputs/cfd2017_single_source_preflight.json")
    p.add_argument("--x-centre", type=float, default=0.010)
    p.add_argument("--z-top", type=float, default=0.006)
    p.add_argument("--y-start", type=float, default=0.008)
    p.add_argument("--speed", type=float, default=0.018)
    p.add_argument("--laser-time", type=float, default=1.5)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--power", type=float, default=0.7 * 3200.0 + 0.3 * 1000.0, help="absorbed power, full plate [W]")
    p.add_argument("--cool-to", type=float, default=60.0)
    p.add_argument("--cool-growth", type=float, default=1.15)
    p.add_argument("--cool-dt-max", type=float, default=2.0)
    a = p.parse_args(argv)

    rows, t, sid = [], 0.0, 0
    n_on = int(round(a.laser_time / a.dt))
    for i in range(n_on):
        t += a.dt
        y = a.y_start + a.speed * (t - 0.5 * a.dt)  # centre at mid-interval
        rows.append([f"{t:.15g}", f"{a.x_centre:.6f}", f"{y:.6f}", f"{a.z_top:.6f}", f"{a.power:.6f}", 1, 1, 1, "weld", f"{a.z_top:.6f}", sid]); sid += 1
    y_end = a.y_start + a.speed * a.laser_time
    dt = a.dt
    while t < a.cool_to - 1e-12:
        dt = min(dt * a.cool_growth, a.cool_dt_max, a.cool_to - t)
        t += dt
        rows.append([f"{t:.15g}", f"{a.x_centre:.6f}", f"{y_end:.6f}", f"{a.z_top:.6f}", "0", 0, 1, 1, "cooling", f"{a.z_top:.6f}", sid]); sid += 1
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "x", "y", "z", "power", "laser_on", "layer", "hatch", "mode", "front_coord", "scan_id"])
        w.writerows(rows)
    rep = dict(rows=len(rows), heating_rows=n_on, heating_energy_J=a.power * a.laser_time, power_W=a.power,
               travel_mm=[a.y_start * 1e3, y_end * 1e3], total_time_s=t, cfd_end_s=1.7,
               notes=["single equivalent source (D-C-01)", "coordinates: mesh metres, weld along y at x_centre",
                      "run with --dt %g --absorptivity 1.0 --recoat-time 0 --cooling-steps 0 --layers 1" % a.dt])
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
