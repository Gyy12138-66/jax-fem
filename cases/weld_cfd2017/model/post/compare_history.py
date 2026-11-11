# -*- coding: utf-8 -*-
"""
比较若干 run 的 result/temp_history.txt（8 个探针点的温度历史）：以第一个 run 为参考，
把其它 run 的曲线插值到参考时刻，给出各探针的最大偏差与出现时刻。
用法: python compare_history.py runs/cool10s runs/cool10s_adapt runs/cool10s_adaptr8 [--tmax 11.5]
"""
import os, sys
import numpy as np


def load(d):
    a = np.loadtxt(os.path.join(d, 'result', 'temp_history.txt'), skiprows=1, ndmin=2)
    # 去重（被拒绝的步不会写，但保险起见按时间排序去重）
    t, idx = np.unique(a[:, 0], return_index=True)
    return t, a[idx, 1:]


def main(argv):
    tmax = None
    if '--tmax' in argv:
        i = argv.index('--tmax'); tmax = float(argv[i + 1]); del argv[i:i + 2]
    dirs = argv
    t0, T0 = load(dirs[0])
    if tmax is not None:
        m = t0 <= tmax + 1e-9; t0, T0 = t0[m], T0[m]
    print('reference %s: %d rows, t=%.3f..%.3f, probes=%d' % (dirs[0], len(t0), t0[0], t0[-1], T0.shape[1]))
    for d in dirs[1:]:
        t1, T1 = load(d)
        lo, hi = max(t0[0], t1[0]), min(t0[-1], t1[-1])
        m = (t0 >= lo) & (t0 <= hi)
        tt = t0[m]
        print('== %s: %d rows, t=%.3f..%.3f, compared on [%.3f, %.3f]' % (d, len(t1), t1[0], t1[-1], lo, hi))
        worst = 0.0
        for p in range(T0.shape[1]):
            Ti = np.interp(tt, t1, T1[:, p])
            dT = Ti - T0[m, p]
            k = int(np.abs(dT).argmax())
            worst = max(worst, abs(dT[k]))
            print('   probe %d: max|dT|=%7.2f K at t=%.3f s (T_ref=%.1f)   rms=%.3f K' % (p + 1, abs(dT[k]), tt[k], T0[m, p][k], np.sqrt((dT ** 2).mean())))
        print('   overall max|dT| = %.2f K' % worst)


if __name__ == '__main__':
    main(sys.argv[1:])
