#!/usr/bin/env python3
"""Map the CFD temperature frames (Desktop/熔池温度场result/output/v01) onto the mechanical mesh nodes.

CFD frame: x along the weld (0..44 mm), y transverse half-model (0..10 mm, symmetry y=0), z thickness (0..6 mm).
Mechanical mesh: --mapping identity expects the CFD axes (x along the weld, y across it with the weld
centreline at --sym-centre, z through thickness); --mapping swap handles the legacy weld-along-y meshes.
Trilinear interpolation on the
rectilinear CFD node grid; temperatures capped at --cap (the CFD surface peaks reach 4000+ K, the
mechanics only needs "above liquidus").

Outputs: npz (time, T[n_frames, n_nodes], points) for --prescribed-temperature-file, the matching
path csv (one runner step per frame, no heat source) and a JSON report.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np


def read_inp_nodes(path):
    ids, xyz = [], []
    with open(path) as f:
        in_nodes = False
        for line in f:
            u = line.strip()
            if u.startswith('*'):
                in_nodes = u.upper().startswith('*NODE')
                continue
            if in_nodes and u:
                parts = u.split(',')
                ids.append(int(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    ids = np.asarray(ids)
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise SystemExit('inp node ids are not consecutive 1..N; node order mapping would be ambiguous')
    return np.asarray(xyz, dtype=np.float64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v01', required=True, help='directory with field_grid.bin/field_T.bin/field_index.txt and tools/')
    p.add_argument('--frames', default=None, help='subset list (frame ids in first column, # comments); default all frames')
    p.add_argument('--inp', required=True)
    p.add_argument('--mapping', choices=('identity', 'swap'), default='identity',
                   help="'identity': the mesh already uses the CFD axes (x along the weld, y across it with the "
                        "weld centreline at --sym-centre, z through thickness). 'swap': legacy meshes whose weld "
                        "runs along y (cfd_x = mesh_y, cfd_y = |mesh_x - --sym-centre|).")
    p.add_argument('--sym-centre', type=float, default=0.0,
                   help='coordinate of the weld centreline on the transverse axis, in mesh coordinates [m]')
    p.add_argument('--x-centre', type=float, default=None, help='deprecated alias of --sym-centre')
    p.add_argument('--cap', type=float, default=1000.0, help='cap nodal temperature at this value [K]')
    p.add_argument('--out-npz', required=True)
    p.add_argument('--out-path', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--laser-time', type=float, default=1.5)
    a = p.parse_args(argv)

    sys.path.insert(0, os.path.join(a.v01, 'tools'))
    from read_field import FieldReader
    from scipy.interpolate import RegularGridInterpolator

    fr = FieldReader(a.v01)
    if a.frames:
        sel = [int(l.split()[0]) for l in open(a.frames, encoding='utf-8') if l.strip() and not l.startswith('#')]
    else:
        sel = list(range(fr.nframe))

    pts = read_inp_nodes(a.inp)
    centre = a.sym_centre if a.x_centre is None else a.x_centre
    if a.mapping == 'identity':
        cfd_pts = np.column_stack([pts[:, 0], np.abs(pts[:, 1] - centre), pts[:, 2]])
    else:
        cfd_pts = np.column_stack([pts[:, 1], np.abs(pts[:, 0] - centre), pts[:, 2]])
    lo = np.array([fr.x[0], fr.y[0], fr.z[0]]); hi = np.array([fr.x[-1], fr.y[-1], fr.z[-1]])
    out_of_box = np.maximum(lo - cfd_pts, cfd_pts - hi).max(axis=1)
    print('mesh nodes %d; mapped coordinate range x %.4f..%.4f y %.4f..%.4f z %.4f..%.4f; max excursion beyond CFD box %.2e m'
          % (len(pts), cfd_pts[:, 0].min(), cfd_pts[:, 0].max(), cfd_pts[:, 1].min(), cfd_pts[:, 1].max(),
             cfd_pts[:, 2].min(), cfd_pts[:, 2].max(), out_of_box.max()))
    if out_of_box.max() > 1e-6:
        raise SystemExit('mechanical mesh extends beyond the CFD domain')

    times, fields, rows, report = [], [], [], []
    for k in sel:
        t, T, fl = fr.load_frame(k)
        interp = RegularGridInterpolator((fr.x, fr.y, fr.z), T, method='linear', bounds_error=False, fill_value=None)
        Tn = interp(cfd_pts)
        Tn_capped = np.minimum(Tn, a.cap)
        n_liq_nodes = int((Tn >= fr.tliquid).sum())
        n_liq_cfd = int((T >= fr.tliquid).sum())
        report.append(dict(frame=k, time=float(t), Tmax_cfd=float(T.max()), Tmax_nodes=float(Tn.max()),
                           Tmin_nodes=float(Tn.min()), nodes_above_liquidus=n_liq_nodes,
                           cfd_nodes_above_liquidus=n_liq_cfd, nodes_capped=int((Tn > a.cap).sum())))
        times.append(float(t)); fields.append(Tn_capped.astype(np.float64))
        src = min(0.008 + 0.018 * t, 0.008 + 0.018 * a.laser_time)
        mode = 'weld' if t <= a.laser_time + 1e-9 else 'cooling'
        pos = (src, centre) if a.mapping == 'identity' else (centre, src)
        rows.append([f'{t:.15g}', f'{pos[0]:.6f}', f'{pos[1]:.6f}', '0.006000', '0', 0, 1, 1, mode, '0.006000', len(rows)])
        print('frame %3d t=%7.3f  Tmax cfd %7.1f -> nodes %7.1f  liquid nodes %6d (cfd %6d)  capped %d'
              % (k, t, T.max(), Tn.max(), n_liq_nodes, n_liq_cfd, int((Tn > a.cap).sum())))

    times = np.asarray(times); Tarr = np.vstack(fields)
    np.savez(a.out_npz, time=times, T=Tarr, points=pts, cap=a.cap, x_centre=a.x_centre,
             source=os.path.abspath(a.v01), frames=np.asarray(sel))
    with open(a.out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time', 'x', 'y', 'z', 'power', 'laser_on', 'layer', 'hatch', 'mode', 'front_coord', 'scan_id'])
        w.writerows(rows)
    with open(a.report, 'w', encoding='utf-8') as f:
        json.dump(dict(inp=os.path.abspath(a.inp), v01=os.path.abspath(a.v01), frames=sel, cap_K=a.cap,
                       sym_centre_m=centre, n_nodes=int(len(pts)), tsolid=float(fr.tsolid), tliquid=float(fr.tliquid),
                       mapping=('cfd_x=mesh_x, cfd_y=|mesh_y-sym_centre|, cfd_z=mesh_z (CFD axes)'
                                if a.mapping == 'identity' else
                                'cfd_x=mesh_y, cfd_y=|mesh_x-sym_centre|, cfd_z=mesh_z (legacy swapped axes)'),
                       per_frame=report), f, indent=1)
    print('wrote %s (%d frames x %d nodes), %s, %s' % (a.out_npz, len(times), len(pts), a.out_path, a.report))


if __name__ == '__main__':
    main()
