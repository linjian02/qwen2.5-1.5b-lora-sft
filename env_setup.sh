#!/usr/bin/env bash
# 训练机环境准备（Linux + NVIDIA GPU）。在你想放项目的目录下执行。
# 用法：bash env_setup.sh
set -euo pipefail

echo "=== 1/4 检查 GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "❌ 没有检测到 NVIDIA GPU。无卡机器请用 AutoDL/蜂鸟等按小时租卡（见 README 第 3 节）"; exit 1; }

echo "=== 2/4 创建 Python 环境 ==="
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 没有 conda。云镜像一般自带；没有就先装 Miniconda："
  echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
  exit 1
fi
conda create -n lf python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lf

echo "=== 3/4 安装 LLaMA-Factory ==="
if [ ! -d "LLaMA-Factory" ]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..

echo "=== 4/4 安装 vLLM（部署用，可选但推荐）==="
pip install "vllm>=0.6.0" || echo "⚠️ vLLM 装失败不影响训练，可以先用 LLaMA-Factory 的 api 起服务"

echo
echo "✅ 环境就绪。接下来："
echo "   conda activate lf"
echo "   llamafactory-cli version"
echo "   python scripts/build_dataset.py --strict"
