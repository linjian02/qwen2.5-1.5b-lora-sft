#!/usr/bin/env bash
# 起两个 OpenAI 兼容服务：base 在 8001，微调后的在 8000。
# 评测要在同一台机器、同一套参数下跑，否则延迟没有可比性。
# 用法：bash scripts/start_services.sh [merged权重目录]
set -euo pipefail

MERGED="${1:-saves/qwen2.5-7b/merged}"
BASE="Qwen/Qwen2.5-7B-Instruct"
mkdir -p out/logs

if [ ! -d "$MERGED" ]; then
  echo "❌ 找不到 $MERGED，先跑：llamafactory-cli export configs/export_lora.yaml"
  exit 1
fi

echo "启动 base（端口 8001）…"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$BASE" --served-model-name qwen2.5-7b-instruct \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
  --port 8001 > out/logs/base.log 2>&1 &

echo "启动 sft（端口 8000）…"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED" --served-model-name qwen-sft \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.45 \
  --port 8000 > out/logs/sft.log 2>&1 &

echo "等待加载（约 1~3 分钟，7B bf16 每个约 15GB 显存）…"
for i in $(seq 1 60); do
  sleep 5
  b=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/v1/models || true)
  s=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models || true)
  echo "  [$((i*5))s] base=$b sft=$s"
  if [ "$b" = "200" ] && [ "$s" = "200" ]; then
    echo "✅ 两个服务都起来了"
    break
  fi
done

echo
echo "如果没有 200，看日志：tail -50 out/logs/base.log"
echo "显存不够就把 --gpu-memory-utilization 调小，或者一次只起一个（先跑 base 再跑 sft）。"
