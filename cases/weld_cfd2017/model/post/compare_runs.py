# -*- coding: utf-8 -*-
"""
比较若干 run 的结束时刻全场 T（result/T_final.bin）与 cooling_log 统计。
用法: python compare_runs.py runs/cool10s runs/cool10s_adapt runs/cool10s_adaptr8
第一个 run 作为参考。
"""
import os, sys
import numpy as np


def read_final(d):
    p = os.path.join(d, 'result', 'T_final.bin')
    with open(p, 'rb') as f:
        ni, nj, nk = np.fromfile(f, dtype='<i4', count=3)
        x = np.fromfile(f, dtype='<f4', count=ni)
        y = np.fromfile(f, dtype='<f4', count=nj)
        z = np.fromfile(f, dtype='<f4', count=nk)
        t = np.fromfile(f, dtype='<f4', count=1)[0]
        n = int(ni) * int(nj) * int(nk)
        T = np.fromfile(f, dtype='<f4', count=n).reshape((ni, nj, nk), order='F')
        fl = np.fromfile(f, dtype='<f4', count=n).reshape((ni, nj, nk), order='F')
    return float(t), T.astype(np.float64), fl, (x, y, z)


def log_stats(d):
    p = os.path.join(d, 'result', 'cooling_log.txt')
    a = np.loadtxt(p, comments='#', ndmin=2)
    steps = a.shape[0]
    iters = a[:, 2].sum()
    wall = a[:, 3].sum()
    n500 = int((a[:, 2] >= 500).sum())
    solid = a[a[:, 6] == 0]
    return dict(steps=steps, iters=int(iters), wall=wall, n_maxit=n500,
                t_end=a[-1, 1], Tmax_end=a[-1, 4], closure_end=a[-1, 17],
                dE=a[-1, 15], Eflux=a[-1, 16],
                solid_it_per_step=(solid[:, 2].mean() if len(solid) else float('nan')),
                solid_wall_per_step=(solid[:, 3].mean() if len(solid) else float('nan')),
                dt_min=np.diff(a[:, 1]).min() if steps > 1 else float('nan'),
                dt_max=np.diff(a[:, 1]).max() if steps > 1 else float('nan'))


def main(dirs):
    ref = None
    for d in dirs:
        st = log_stats(d)
        print('== %s' % d)
        print('   steps=%d  iters=%d  wall=%.1f s  steps@maxit=%d  dt=[%.4f,%.4f] s' %
              (st['steps'], st['iters'], st['wall'], st['n_maxit'], st['dt_min'], st['dt_max']))
        print('   t_end=%.3f s  Tmax_end=%.2f K  dE_domain=%.3f J  E_flux=%.3f J  closure_rel=%.2e' %
              (st['t_end'], st['Tmax_end'], st['dE'], st['Eflux'], st['closure_end']))
        print('   solid phase: %.1f it/step, %.3f s/step' % (st['solid_it_per_step'], st['solid_wall_per_step']))
        try:
            t, T, fl, g = read_final(d)
        except Exception as e:
            print('   (no T_final.bin: %s)' % e)
            continue
        print('   T_final: t=%.4f  Tmin=%.2f Tmax=%.2f mean=%.2f  liquid nodes=%d' % (t, T.min(), T.max(), T.mean(), int((fl > 0).sum())))
        if ref is None:
            ref = (t, T)
        else:
            dT = T - ref[1]
            print('   vs %s: max|dT|=%.3f K  rms=%.4f K  (t_ref=%.3f, t=%.3f)' % (dirs[0], np.abs(dT).max(), np.sqrt((dT ** 2).mean()), ref[0], t))


if __name__ == '__main__':
    main(sys.argv[1:])
