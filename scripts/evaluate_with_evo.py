#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse

def run_evo(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="用 EVO 分析 TUM 轨迹，生成 APE/RPE 绘图并保存")
    parser.add_argument('--gt_tum',  required=True, help="真值 TUM 文件路径")
    parser.add_argument('--est_tum', required=True, help="估计 TUM 文件路径")
    parser.add_argument('--output_dir', default="evo_results",
                        help="结果图片保存目录")
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # 1) 绝对位姿误差 APE
    cmd_ape = [
        "evo_ape", "tum",
        args.gt_tum, args.est_tum,
        "-v",            # verbose 输出统计信息
        "-a",            # 使用 Umeyama 对齐
        "--plot",        # 显示并绘图
        "--save_plot", os.path.join(out, "ape.png")
    ]
    run_evo(cmd_ape)

    # 2) 相对位姿误差 RPE
    cmd_rpe = [
        "evo_rpe", "tum",
        args.gt_tum, args.est_tum,
        "-v",
        "-a",
        "--plot",
        "--save_plot", os.path.join(out, "rpe.png")
    ]
    run_evo(cmd_rpe)

    print(f"✔ 完成！APE/RPE 图保存在：{os.path.abspath(out)}")

if __name__ == "__main__":
    main()
