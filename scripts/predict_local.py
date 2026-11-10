#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_local.py —— 本机直接加载模型答题（不需要起服务）

为什么单独有这个脚本：
    predict.py 是走 HTTP 接口的，适合服务器上用 vLLM。
    但 vLLM 在 Windows 上装不了，而且 8G 卡同时起两个服务会爆显存。
    这个脚本用 transformers 直接加载模型，一条命令就跑完，最适合单机。

用法（base 与 sft 各跑一次，只差一个 --adapter）：

  # 1) 原始模型（基线）
  python scripts/predict_local.py --model D:/ai/models/Qwen2.5-1.5B-Instruct \
      --out out/pred_base.jsonl --tag base

  # 2) 加了 LoRA 补丁的模型
  python scripts/predict_local.py --model D:/ai/models/Qwen2.5-1.5B-Instruct \
      --adapter D:/ai/kit/saves/qwen2.5-1.5b/lora/sft \
      --out out/pred_sft.jsonl --tag sft

延迟逐条记录，score.py 会算 P50/P99。
"""
import argparse
import json
import os
import sys
import time

SYSTEM_PROMPT = ("你是运营商权益营销话术助手。请根据【检索素材】生成一段可直接发给用户的话术。"
                 "要求：严格按【开场】【核心权益】【办理引导】【限制说明】四段输出；"
                 "只能使用素材中出现过的数字与条件，素材里没有的信息一律不许补；"
                 "语气口语化，不使用「亲」「宝子」等词。")


def load_rows(path):
    rows = []
    if path.endswith(".csv"):
        import csv
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("query") or "").strip():
                    rows.append(r)
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录")
    ap.add_argument("--adapter", default=None, help="LoRA 补丁目录（跑基线时不填）")
    ap.add_argument("--in", dest="inp", default="data/eval/evalset.jsonl")
    ap.add_argument("--out", default="out/pred.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("[X] CUDA 不可用，检查是不是装了 CPU 版 torch")
        sys.exit(1)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"加载分词器：{args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"加载模型（{args.dtype}），首次需要 30~60 秒 ...")
    t_load = time.time()
    # transformers 5.x 把 torch_dtype 改名成 dtype，4.x 只认 torch_dtype。
    # 这里做兼容，避免你看到 "[transformers] `torch_dtype` is deprecated" 这种噪音。
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map={"": 0}, trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map={"": 0}, trust_remote_code=True,
        )
    model.eval()

    if args.adapter:
        from peft import PeftModel
        print(f"挂载 LoRA 补丁：{args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    used = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"加载完成，耗时 {time.time() - t_load:.0f}s，显存占用约 {used:.2f} GB")

    rows = load_rows(args.inp)
    if args.limit:
        rows = rows[: args.limit]
    print(f"待答题：{len(rows)} 条 → {args.out}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
    t_all = time.time()
    for i, row in enumerate(rows, 1):
        query = row.get("query") or row.get("input") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        # apply_chat_template 会自动套上 Qwen 的 <|im_start|> 格式，
        # 手写这个格式是新手最容易出错的地方之一
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # 评测必须贪心解码，保证可复现
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        lat = (time.time() - t0) * 1000
        gen = out[0][inputs["input_ids"].shape[1]:]
        pred = tok.decode(gen, skip_special_tokens=True).strip()

        results.append({
            "query": query, "pred": pred, "latency_ms": round(lat, 1),
            "tag": args.tag, "gold": row.get("answer", ""),
            "difficulty": row.get("difficulty", ""), "category": row.get("category", ""),
            "error": None,
        })
        if i % 5 == 0 or i == len(rows):
            print(f"  进度 {i}/{len(rows)}  末条 {lat:.0f} ms", end="\r", flush=True)

    print()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lats = [r["latency_ms"] for r in results]
    print(f"完成：{len(results)} 条，总耗时 {time.time() - t_all:.0f}s，"
          f"平均 {sum(lats) / len(lats):.0f} ms → {args.out}")


if __name__ == "__main__":
    main()
