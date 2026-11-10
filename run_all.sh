#!/usr/bin/env bash
# 一键跑完「校验 → 构建 → 切留出集 → 训练 → 评测」，Linux / macOS 用。
# Windows 用 docs/本地操作手册.md 里的 6 个 bat 脚本，效果等价。
#
# 用法：bash run_all.sh [留出集条数，默认 100]
set -euo pipefail

N_EVAL="${1:-100}"

echo "=== 1/6 事实溯源校验（6 项） ==="
python scripts/check_rules.py

echo
echo "=== 2/6 构建训练集（格式检查 + SimHash 去重） ==="
python scripts/build_dataset.py

echo
echo "=== 3/6 切留出集（必须在训练之前） ==="
python scripts/make_evalset.py split --n "$N_EVAL"
python scripts/make_evalset.py check

echo
echo "=== 4/6 训练 ==="
mkdir -p out/logs
llamafactory-cli train configs/train_lora_1.5b.yaml 2>&1 | tee out/logs/train.log
echo
grep -E "trainable params|'train_loss'|'eval_loss'" out/logs/train.log | tail -5 || true

echo
echo "=== 5/6 基座与微调模型各答一遍 ==="
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
python scripts/predict_local.py --model "$MODEL" --out out/pred_base.jsonl --tag base
python scripts/predict_local.py --model "$MODEL" \
       --adapter saves/qwen2.5-1.5b/lora/sft --out out/pred_sft.jsonl --tag sft

echo
echo "=== 6/6 评测 ==="
python scripts/score.py --pred out/pred_base.jsonl out/pred_sft.jsonl

echo
echo "完成。报告见 out/score_report.md"
