# -*- coding: utf-8 -*-
"""实验七 步骤3：一键流水线 —— Blender 导出控制图 → sd-cli 纯 CPU 生成写实图
用法: python scripts/run_pipeline.py --name test512 --res 512 --steps 4 --cfg 1.2
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER_EXE = os.environ.get("BLENDER_EXE", "/Applications/Blender.app/Contents/MacOS/Blender")
SDCLI = ROOT / "vendor" / "sdcpp-ext" / "sd-cli"
MODEL_SD = ROOT / "models" / "sd_turbo_q4_0.gguf"
CONTROLNET = ROOT / "models" / "control_depth_sd21-fp16.gguf"

PROMPT = ("photorealistic render, studio soft lighting, PBR material, "
          "preserve original 3d geometry, sharp")
NEGATIVE = "deformed geometry, perspective changed"


def ensure_rgb(path):
    """sd.cpp 的 control image 需要至少 3 通道。"""
    from PIL import Image
    with Image.open(path) as img:
        if img.mode != "RGB":
            img.convert("RGB").save(path)


def latest(pattern):
    files = glob.glob(str(pattern))
    return Path(max(files, key=os.path.getmtime)) if files else None


def parse_sdcli_timing(log):
    """从 sd-cli 输出解析采样与 VAE 耗时（格式因版本而异，做宽松匹配）。"""
    sample = vae = None
    for m in re.finditer(r"sample.*?time\s*=\s*([\d.]+)s", log, re.I):
        sample = float(m.group(1))
    for m in re.finditer(r"vae.*?time\s*=\s*([\d.]+)s", log, re.I):
        vae = float(m.group(1))
    return sample, vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="test512")
    ap.add_argument("--res", type=int, default=512, choices=(512, 768))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--control-strength", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--control-image", default=None,
                    help="跳过 Blender，直接使用已有深度图（绝对路径）")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--blend", default=str(ROOT / "assets" / "demo_white_model.blend"))
    args = ap.parse_args()

    out_root = ROOT / "output"
    out_root.mkdir(exist_ok=True)
    t_total0 = time.time()

    # ---------- 1) Blender 导出控制图 ----------
    t_blender0 = time.time()
    if args.control_image:
        depth_path = Path(args.control_image)
        print(f"[pipeline] 跳过 Blender，使用已有深度图 {depth_path}")
    else:
        export_script = ROOT / "blender" / "export_control_maps.py"
        r = subprocess.run(
            [BLENDER_EXE, "-b", args.blend, "-P", str(export_script), "--",
             "--outdir", str(out_root), "--name", args.name, "--res", str(args.res)],
            capture_output=True, text=True)
        print(r.stdout[-500:])
        if r.returncode != 0:
            print(r.stderr[-800:])
            raise SystemExit("Blender 导出失败")
        depth_path = out_root / f"{args.name}_depth.png"
        if not depth_path.exists():
            src = latest(out_root / f"{args.name}_depth_*.png")
            if src is None:
                raise SystemExit("未找到 depth 输出文件")
            shutil.copyfile(src, depth_path)
        ensure_rgb(depth_path)
    t_blender = time.time() - t_blender0
    print(f"[pipeline] Blender 导出耗时 {t_blender:.1f} s")

    # ---------- 2) sd-cli 生成 ----------
    photo_path = out_root / f"{args.name}_photoreal.png"
    sd_cmd = [
        str(SDCLI),
        "-m", str(MODEL_SD),
        "-p", args.prompt,
        "-n", args.negative,
        "--control-net", str(CONTROLNET),
        "--control-image", str(depth_path),
        "--control-strength", str(args.control_strength),
        "-W", str(args.res), "-H", str(args.res),
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg),
        "--sampling-method", "euler_a",
        "-s", str(args.seed),
        "-t", str(args.threads),
        "-o", str(photo_path),
        "-v",
    ]
    print(f"[pipeline] sd-cli: {' '.join(sd_cmd[1:8])} ...")
    t_sd0 = time.time()
    r = subprocess.run(sd_cmd, cwd=SDCLI.parent, capture_output=True, text=True)
    t_sd = time.time() - t_sd0
    log = (r.stdout or "") + "\n" + (r.stderr or "")
    with open(out_root / f"{args.name}_sdcli.log", "w") as f:
        f.write(log)
    if r.returncode != 0 or not photo_path.exists():
        print(log[-1500:])
        raise SystemExit("sd-cli 生成失败")
    sample_t, vae_t = parse_sdcli_timing(log)

    # ---------- 3) 耗时汇总 ----------
    total = time.time() - t_total0
    print("=" * 60)
    print(f"运行 {args.name}  分辨率 {args.res}²  步数 {args.steps}  CFG {args.cfg}")
    print(f"  Blender 导出: {t_blender:.1f} s")
    print(f"  sd-cli 总耗时: {t_sd:.1f} s（采样 {sample_t if sample_t else '?'} s, "
          f"VAE {vae_t if vae_t else '?'} s）")
    print(f"  总耗时: {total:.1f} s")
    print(f"  输出: {photo_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
