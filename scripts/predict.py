#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py —— 对评测集批量推理，记录输出与延迟

只要服务是 OpenAI 兼容接口就能用（vLLM / LLaMA-Factory api / Ollama / 任何云 API）。
零第三方依赖（urllib + 线程池），不用装 requests / openai。

用法（base 与 sft 各跑一次，tag 不同）：
  python scripts/predict.py --base-url http://127.0.0.1:8001/v1 --model qwen2.5-7b-instruct \
      --in data/eval/evalset.jsonl --out out/pred_base.jsonl --tag base

  python scripts/predict.py --base-url http://127.0.0.1:8000/v1 --model qwen-sft \
      --in data/eval/evalset.jsonl --out out/pred_sft.jsonl --tag sft

延迟是逐条记的（latency_ms），后面 score.py 会算 P50/P99 —— 这是简历上「P99 延迟」的来源。
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SYSTEM_PROMPT = ("你是运营商权益营销话术助手。请根据【检索素材】生成一段可直接发给用户的话术。"
                 "要求：严格按【开场】【核心权益】【办理引导】【限制说明】四段输出；"
                 "只能使用素材中出现过的数字与条件，素材里没有的信息一律不许补；"
                 "语气口语化，不使用「亲」「宝子」等词。")

SECTIONS = ["【开场】", "【核心权益】", "【办理引导】", "【限制说明】"]


def call(base_url, model, prompt, temperature, max_tokens, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": [],
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    lat = (time.time() - t0) * 1000
    text = payload["choices"][0]["message"]["content"].strip()
    return text, lat


def build_prompt(row):
    q = row.get("query") or row.get("input") or ""
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", default="data/eval/evalset.jsonl")
    ap.add_argument("--out", default="out/pred.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="评测固定 0，保证可复现；不要用 0.7 再怪模型不稳定")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    if args.inp.endswith(".csv"):
        import csv
        with open(args.inp, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("query") or "").strip():
                    rows.append({"query": r["query"], "answer": r.get("answer", ""),
                                 "difficulty": r.get("difficulty", ""), "category": r.get("category", "")})
    else:
        with open(args.inp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if args.limit:
        rows = rows[: args.limit]
    print(f"待推理：{len(rows)} 条 → {args.base_url}  model={args.model}  concurrency={args.concurrency}")

    lock = threading.Lock()
    done = [0]
    results = [None] * len(rows)

    def work(i, row):
        prompt = build_prompt(row)
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                text, lat = call(args.base_url, args.model, prompt,
                                 args.temperature, args.max_tokens, args.timeout)
                return i, {"query": prompt, "pred": text, "latency_ms": round(lat, 1),
                           "tag": args.tag, "gold": row.get("answer", ""),
                           "difficulty": row.get("difficulty", ""),
                           "category": row.get("category", ""), "error": None}
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1 + attempt * 2)
        return i, {"query": prompt, "pred": "", "latency_ms": None, "tag": args.tag,
                   "gold": row.get("answer", ""), "difficulty": row.get("difficulty", ""),
                   "category": row.get("category", ""), "error": str(last_err)[:200]}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, i, r) for i, r in enumerate(rows)]
        for fu in as_completed(futs):
            i, rec = fu.result()
            results[i] = rec
            with lock:
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(rows):
                    print(f"  进度 {done[0]}/{len(rows)}", end="\r", flush=True)

    print()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errs = sum(1 for r in results if r["error"])
    print(f"完成：{len(results)} 条，失败 {errs} 条 → {args.out}")
    if errs:
        print(f"首条失败原因：{next(r['error'] for r in results if r['error'])}")
        print("排查顺序：服务是否起来（curl 一下 /v1/models）→ model 名是否写对 → 端口是否正确")
        sys.exit(2)


if __name__ == "__main__":
    main()
