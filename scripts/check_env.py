#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境自检：一条命令告诉你"现在能不能开始训练"。

用法：
    python scripts/check_env.py

它只读不写，跑完不会改动任何东西。报错就按最后「结论」下面的提示做。
"""
import os
import shutil
import subprocess
import sys

OK = "[OK]"
NO = "[X] "
WARN = "[!] "


def line(t=""):
    print(t)


def hr(title):
    line()
    line("=" * 62)
    line(title)
    line("=" * 62)


def check_gpu():
    hr("1) 显卡")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0 or not out.stdout.strip():
            line(NO + "nvidia-smi 没有返回信息，可能没装 NVIDIA 驱动")
            return None
        rows = [r for r in out.stdout.strip().splitlines() if r.strip()]
        total_mb = 0
        for r in rows:
            parts = [p.strip() for p in r.split(",")]
            name, mem, drv = parts[0], parts[1], parts[2]
            mb = int("".join(ch for ch in mem if ch.isdigit()))
            total_mb = max(total_mb, mb)
            line(f"{OK} {name} / 显存 {mb} MiB / 驱动 {drv}")
        if total_mb < 7000:
            line(WARN + f"显存只有 {total_mb} MiB，建议用 1.5B 配方，不要上 3B")
        elif total_mb < 20000:
            line(OK + "显存够跑 1.5B（不量化）；3B 需要 4bit 量化")
        else:
            line(OK + "显存 ≥20G，可以直接用 7B 配方")
        return total_mb
    except FileNotFoundError:
        line(NO + "找不到 nvidia-smi。请确认装的是 NVIDIA 显卡 + 官方驱动")
        return None
    except Exception as e:
        line(NO + f"检测显卡出错：{e}")
        return None


def check_python():
    hr("2) Python 与依赖")
    v = sys.version_info
    line(f"{OK if v >= (3, 10) else NO} Python {v.major}.{v.minor}.{v.micro}")
    line(f"     解释器：{sys.executable}")
    if v < (3, 10):
        line(NO + "版本过低，LLaMA-Factory 需要 3.10 或 3.11")
    elif v >= (3, 13):
        line(WARN + "版本偏新，部分依赖可能没有对应轮子，建议 3.10/3.11")

    # torch
    try:
        import torch
        line(f"{OK} torch {torch.__version__}")
        if torch.cuda.is_available():
            line(OK + f"CUDA 可用，设备：{torch.cuda.get_device_name(0)}")
            line(f"     CUDA 版本：{torch.version.cuda}")
            try:
                props = torch.cuda.get_device_properties(0)
                gb = props.total_memory / 1024 ** 3
                line(f"     可用显存：{gb:.1f} GB，计算能力 sm_{props.major}{props.minor}")
                if props.major == 8 and props.minor == 6:
                    line(OK + "sm_86（30 系），支持 bf16，配方里的 bf16: true 可用")
            except Exception:
                pass
            try:
                line(f"     bf16 支持：{torch.cuda.is_bf16_supported()}")
            except Exception:
                pass
        else:
            line(NO + "torch 装上了但 CUDA 不可用 —— 大概率装成了 CPU 版")
            line("     重装：pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124")
    except ImportError:
        line(NO + "没装 torch")
        line("     双击 D:\\ai\\0_一键安装.bat，或看 0_新手必读.md 第四节")

    # 训练依赖
    need = ["transformers", "datasets", "accelerate", "peft", "trl", "safetensors", "yaml"]
    missing = []
    for m in need:
        try:
            mod = __import__(m)
            ver = getattr(mod, "__version__", "")
            line(f"{OK} {m} {ver}")
        except ImportError:
            missing.append(m)
            line(NO + f"缺 {m}")
    if missing:
        line("     双击 D:////ai////0_一键安装.bat 补齐")

    # llamafactory
    try:
        import llamafactory
        line(OK + " llamafactory 已安装")
    except ImportError:
        line(NO + "没装 llamafactory，训练命令 llamafactory-cli 会找不到")
        line("     双击 D:\\ai\\0_一键安装.bat 补齐")

    # 量化库（只有 3B 配方需要）
    try:
        import bitsandbytes
        line(OK + f" bitsandbytes {getattr(bitsandbytes, '__version__', '')}（3B 4bit 需要）")
    except ImportError:
        line(WARN + "没装 bitsandbytes —— 只影响 3B QLoRA，跑 1.5B 不需要它")

    return not missing and sys.version_info >= (3, 10)


def check_model():
    hr("3) 模型文件")
    base = r"D:/ai/models"
    if not os.path.isdir(base):
        line(NO + f"没有 {base} 目录")
        return False
    found = False
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        files = os.listdir(p)
        has_cfg = "config.json" in files
        weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
        size_gb = 0
        for w in weights:
            try:
                size_gb += os.path.getsize(os.path.join(p, w)) / 1024 ** 3
            except OSError:
                pass
        if has_cfg and weights:
            line(f"{OK} {name}：{len(weights)} 个权重文件，合计 {size_gb:.1f} GB")
            found = True
        else:
            line(WARN + f"{name}：文件不全（缺 config.json 或权重）")
    if not found:
        line(NO + "没找到完整模型")
        line("     双击 D:////ai////0_一键安装.bat，会自动下载模型")
    return found


def check_data():
    hr("4) 数据")
    kit = r"D:/ai/kit"
    ok = True
    seed = os.path.join(kit, "data", "seed", "seed33.jsonl")
    if os.path.exists(seed):
        n = sum(1 for _ in open(seed, encoding="utf-8") if _.strip())
        line(f"{OK} seed33.jsonl：{n} 条")
        if n < 30:
            line(WARN + "条数偏少，建议至少 100 条再训练")
    else:
        line(NO + "找不到 data/seed/seed33.jsonl")
        ok = False
    info = os.path.join(kit, "data", "dataset_info.json")
    if os.path.exists(info):
        line(OK + " dataset_info.json 存在")
    else:
        line(NO + "找不到 data/dataset_info.json（LLaMA-Factory 靠它找数据）")
        ok = False
    train_json = os.path.join(kit, "data", "train.json")
    if os.path.exists(train_json):
        line(OK + " data/train.json 已生成（跑过质检）")
    else:
        line(WARN + "还没生成 train.json，训练前先跑：python scripts/build_dataset.py")
    return ok


def check_disk():
    hr("5) 磁盘")
    for drive in ["C:\\", "D:\\"]:
        try:
            total, used, free = shutil.disk_usage(drive)
            gb = free / 1024 ** 3
            flag = OK if gb > 10 else (WARN if gb > 3 else NO)
            line(f"{flag} {drive} 剩余 {gb:.0f} GB")
        except Exception:
            pass
    line("     训练产物会写到 D:\\ai\\kit\\saves，请保证 D 盘有 10G 以上空余")
    return True


def main():
    print()
    print("环境自检（只读，不会改动任何文件）")
    line(f"工作目录：{os.getcwd()}")
    gpu = check_gpu()
    dep_ok = check_python()
    model_ok = check_model()
    data_ok = check_data()
    check_disk()

    hr("结论")
    if gpu is None:
        line(NO + "显卡没识别到，先解决驱动问题，其他都不用管")
    elif not dep_ok:
        line(NO + "显卡没问题，但 Python 依赖没装全，按上面 [X] 的提示补齐")
    elif not model_ok:
        line(NO + "依赖齐了，还差模型文件，按步骤 5 下载")
    elif not data_ok:
        line(NO + "环境和模型都好了，数据有问题，看第 4 节提示")
    else:
        line(OK + "环境可用，可以开始训练")
        line("     python scripts/build_dataset.py")
        line("     llamafactory-cli train configs/train_lora_1.5b.yaml")
    line()


if __name__ == "__main__":
    main()
