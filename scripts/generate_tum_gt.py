#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def load_matrices(mat_path):
    """
    每行 12 个数，按行主序 [r00 r01 r02 t0 | r10 r11 r12 t1 | r20 r21 r22 t2]
    返回 N×3×3 的 R_list 和 N×3 的 t_list
    """
    data = np.loadtxt(mat_path)
    assert data.ndim == 2 and data.shape[1] == 12
    Rs = data[:, [0,1,2, 4,5,6, 8,9,10]].reshape(-1, 3, 3)
    ts = data[:, [3,7,11]]
    return Rs, ts

def load_timestamps(ts_path):
    """
    每行一个浮点数，单位同你算法输出（秒）
    """
    return np.loadtxt(ts_path)

def write_tum(gt_path, timestamps, Rs, ts):
    """
    生成 TUM 格式：timestamp tx ty tz qx qy qz qw
    """
    assert len(timestamps) == len(Rs) == len(ts)
    with open(gt_path, 'w') as f:
        for t, Rm, tv in zip(timestamps, Rs, ts):
            q = R.from_matrix(Rm).as_quat()  # [x, y, z, w]
            f.write(f"{t:.6f} "
                    f"{tv[0]:.6f} {tv[1]:.6f} {tv[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

def main():
    p = argparse.ArgumentParser(
        description="把 12 列真值矩阵 + 时间戳合成 TUM 格式轨迹")
    p.add_argument('--matrix_file', required=True,
                   help="含 12 列的真值文件路径")
    p.add_argument('--timestamp_file', required=True,
                   help="对应行数的时间戳文件路径")
    p.add_argument('--output_tum', default="groundtruth.tum",
                   help="输出 TUM 文件名")
    args = p.parse_args()

    Rs, ts = load_matrices(args.matrix_file)
    times = load_timestamps(args.timestamp_file)
    write_tum(args.output_tum, times, Rs, ts)
    print(f"✔ 已写入 TUM 真值：{args.output_tum}")

if __name__ == "__main__":
    main()