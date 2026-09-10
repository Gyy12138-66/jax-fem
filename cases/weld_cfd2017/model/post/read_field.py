# -*- coding: utf-8 -*-
"""
读取 cooling_v1 输出的全场二进制温度帧，供应力计算做插值映射。

文件（均在 run 目录的 result/ 下）：
  field_grid.bin  : int32 ni,nj,nk ; float32 x(ni),y(nj),z(nk) [m] ; float32 tsolid,tliquid
  field_T.bin     : 逐帧追加  float32 time ; float32 T(ni,nj,nk) ; float32 fracl(ni,nj,nk)  （Fortran 列优先）
  field_index.txt : frame time byte_offset(0-based) Tmax nliq

用法：
  python read_field.py <result_dir>                 # 打印帧表并做一致性自检
  python read_field.py <result_dir> --probe 0.030 0.0 0.006 --t 2.0   # 在 (x,y,z) [m]、t [s] 处插值取温度

作为库：
  from read_field import FieldReader
  fr = FieldReader('runs/cool10s_adapt/result')
  T = fr.temperature_at(points_xyz, t)   # points_xyz: (N,3) [m]，半对称模型内部对 y 取 |y|
"""
import os, sys
import numpy as np


class FieldReader:
    def __init__(self, result_dir):
        self.dir = result_dir
        g = open(os.path.join(result_dir, 'field_grid.bin'), 'rb')
        ni, nj, nk = np.fromfile(g, dtype='<i4', count=3)
        self.ni, self.nj, self.nk = int(ni), int(nj), int(nk)
        self.x = np.fromfile(g, dtype='<f4', count=self.ni).astype(np.float64)
        self.y = np.fromfile(g, dtype='<f4', count=self.nj).astype(np.float64)
        self.z = np.fromfile(g, dtype='<f4', count=self.nk).astype(np.float64)
        self.tsolid, self.tliquid = np.fromfile(g, dtype='<f4', count=2)
        g.close()
        idx = np.loadtxt(os.path.join(result_dir, 'field_index.txt'), comments='#', ndmin=2)
        self.frame_time = idx[:, 1]
        self.frame_offset = idx[:, 2].astype(np.int64)
        self.frame_tmax = idx[:, 3]
        self.frame_nliq = idx[:, 4].astype(int)
        self.nframe = len(self.frame_time)
        self.npt = self.ni * self.nj * self.nk
        self.path_T = os.path.join(result_dir, 'field_T.bin')
        self._cache = {}

    def load_frame(self, k):
        """返回 (time, T[ni,nj,nk], fracl[ni,nj,nk])，数组按 (i,j,k) 索引。"""
        if k in self._cache:
            return self._cache[k]
        with open(self.path_T, 'rb') as f:
            f.seek(int(self.frame_offset[k]))
            t = np.fromfile(f, dtype='<f4', count=1)[0]
            T = np.fromfile(f, dtype='<f4', count=self.npt).reshape((self.ni, self.nj, self.nk), order='F')
            fl = np.fromfile(f, dtype='<f4', count=self.npt).reshape((self.ni, self.nj, self.nk), order='F')
        out = (float(t), T.astype(np.float64), fl.astype(np.float64))
        if len(self._cache) > 4:
            self._cache.pop(next(iter(self._cache)))
        self._cache[k] = out
        return out

    def field_at_time(self, t):
        """帧间线性插值得到 t 时刻的 T 场（越界取端点帧）。"""
        ft = self.frame_time
        if t <= ft[0]:
            return self.load_frame(0)[1]
        if t >= ft[-1]:
            return self.load_frame(self.nframe - 1)[1]
        k = int(np.searchsorted(ft, t, side='right') - 1)
        t0, T0, _ = self.load_frame(k)
        t1, T1, _ = self.load_frame(k + 1)
        w = (t - t0) / (t1 - t0)
        return (1.0 - w) * T0 + w * T1

    def temperature_at(self, pts, t, cap_to_liquidus=False):
        """在空间点 pts (N,3) [m] 与时刻 t [s] 插值温度。半对称：y 取绝对值。"""
        from scipy.interpolate import RegularGridInterpolator
        T = self.field_at_time(t)
        if cap_to_liquidus:
            T = np.minimum(T, self.tliquid)
        f = RegularGridInterpolator((self.x, self.y, self.z), T, bounds_error=False, fill_value=None)
        p = np.asarray(pts, dtype=np.float64).copy()
        p[:, 1] = np.abs(p[:, 1])
        return f(p)

    def summary(self):
        print('grid ni,nj,nk = %d %d %d  (%d nodes)' % (self.ni, self.nj, self.nk, self.npt))
        print('x: %.4f .. %.4f m   y: %.4f .. %.4f m   z: %.4f .. %.4f m' %
              (self.x[0], self.x[-1], self.y[0], self.y[-1], self.z[0], self.z[-1]))
        print('tsolid=%.1f K  tliquid=%.1f K' % (self.tsolid, self.tliquid))
        print('frames: %d   t = %.4f .. %.4f s' % (self.nframe, self.frame_time[0], self.frame_time[-1]))
        size = os.path.getsize(self.path_T)
        expect = self.nframe * (4 + 8 * self.npt)
        print('field_T.bin size %d bytes, expected %d -> %s' % (size, expect, 'OK' if size == expect else 'MISMATCH'))
        # 自检：每帧的 Tmax 与索引一致
        bad = 0
        for k in range(self.nframe):
            t, T, fl = self.load_frame(k)
            if abs(t - self.frame_time[k]) > 1e-4 or abs(T.max() - self.frame_tmax[k]) > 0.05:
                bad += 1
        print('per-frame consistency check: %d bad of %d' % (bad, self.nframe))
        # 相邻帧最大温变（衡量时间分辩率是否足够）
        dmax = []
        for k in range(1, self.nframe):
            dmax.append(np.abs(self.load_frame(k)[1] - self.load_frame(k - 1)[1]).max())
        if dmax:
            print('max |dT| between consecutive frames: max %.1f K, median %.1f K' % (max(dmax), float(np.median(dmax))))


if __name__ == '__main__':
    d = sys.argv[1]
    fr = FieldReader(d)
    fr.summary()
    if '--probe' in sys.argv:
        i = sys.argv.index('--probe')
        p = np.array([[float(sys.argv[i + 1]), float(sys.argv[i + 2]), float(sys.argv[i + 3])]])
        t = float(sys.argv[sys.argv.index('--t') + 1])
        print('T(x=%.4f,y=%.4f,z=%.4f, t=%.3f) = %.2f K' % (p[0, 0], p[0, 1], p[0, 2], t, fr.temperature_at(p, t)[0]))
