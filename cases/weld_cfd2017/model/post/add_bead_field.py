# -*- coding: utf-8 -*-
"""给已有的 VTU 序列补上 bead 单元场（1=焊道, 0=母板），按母板单元数分界。"""
import glob, os, re, sys
import numpy as np, meshio
D, n_plate = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(D, "step_*.vtu")), key=lambda p: int(re.search(r"step_(\d+)_", p).group(1)))
done = 0
for f in files:
    m = meshio.read(f)
    if "bead" in m.cell_data:
        continue
    n = len(m.cells[0].data)
    bead = np.zeros(n, dtype=np.float64); bead[n_plate:] = 1.0
    m.cell_data["bead"] = [bead]
    meshio.write(f, m)
    done += 1
print("补写 %d / %d 帧；焊道单元 %d, 母板 %d" % (done, len(files), n - n_plate, n_plate))
