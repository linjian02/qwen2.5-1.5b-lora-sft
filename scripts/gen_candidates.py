#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_candidates.py —— 用基座模型先批量生成数据候选，人工再审

为什么需要它：300 条全靠手写大约 15 小时，一周做不完。
实际做法是「少量手写种子 + 模型批量生成候选 + 人工过一遍」，
这也是工业界常见的 self-instruct 式数据合成，面试讲得清楚反而是加分项。

关键纪律：**生成的候选必须人工过一遍才能入库**。模型编出来的数字如果原样进了训练集，
你会训出一个「更擅长编数字」的模型，评测指标反而更差。

用法：
  1) 准备 topics 文件（只要问题，不要答案）：
       CSV 列：query          —— 每行一条「用户情况 + 检索素材」
       或 JSONL：{"query": "..."}
  2) 跑：
     python scripts/gen_candidates.py --topics data/raw/topics.csv \
         --base-url http://127.0.0.1:8001/v1 --model qwen2.5-7b-instruct \
         --out data/raw/candidates.jsonl --n-shot 3
  3) 打开 out/candidates_review.csv，人工改完再另存为 data/raw/approved.jsonl
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import SYSTEM_PROMPT  # noqa: E402
from build_dataset import section_check  # noqa: E402


def load_rows(path):
    if path.endswith(".csv"):
        out = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                q = (r.get("query") or r.get("问题") or "").strip()
                if q:
                    out.append({"query": q})
        return out
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                q = (d.get("query") or d.get("input") or "").strip()
                if q:
                    out.append({"query": q})
    return out


def load_shots(path, k):
    shots = []
    if not os.path.exists(path):
        return shots
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            shots.append((d.get("input") or d.get("query") or "", d["output"]))
            if len(shots) >= k:
                break
    return shots


def call(base_url, model, messages, timeout, temperature):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 512}
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--seeds", default="data/seed/sample_sft.jsonl")
    ap.add_argument("--n-shot", type=int, default=3)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--out", default="data/raw/candidates.jsonl")
    ap.add_argument("--review", default="out/candidates_review.csv")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="生成候选时用 0.7 提高多样性，和评测时的 0 不一样")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args()

    topics = load_rows(args.topics)
    shots = load_shots(args.seeds, args.n_shot)
    if not shots:
        print(f"没读到 few-shot 示例（{args.seeds}）。至少需要 2~3 条手写样本，否则生成质量很差。")
        sys.exit(1)
    print(f"话题 {len(topics)} 条，few-shot {len(shots)} 条")

    def build_messages(q):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for iq, ia in shots:
            msgs.append({"role": "user", "content": iq})
            msgs.append({"role": "assistant", "content": ia})
        msgs.append({"role": "user", "content": q})
        return msgs

    results = [None] * len(topics)

    def work(i, row):
        for attempt in range(3):
            try:
                txt = call(args.base_url, args.model, build_messages(row["query"]),
                           args.timeout, args.temperature)
                ok, missing, in_order = section_check(txt)
                return i, {"query": row["query"], "output": txt,
                           "_format_ok": ok, "_missing": missing, "_in_order": in_order}
            except Exception as e:  # noqa: BLE001
                err = str(e)[:150]
                time.sleep(1 + attempt * 2)
        return i, {"query": row["query"], "output": "", "_format_ok": False, "_missing": ["请求失败"],
                   "_in_order": False, "_err": err}

    done = [0]
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fu in as_completed([ex.submit(work, i, r) for i, r in enumerate(topics)]):
            i, rec = fu.result()
            results[i] = rec
            done[0] += 1
            print(f"  进度 {done[0]}/{len(topics)}", end="\r", flush=True)
    print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_n = sum(1 for r in results if r["_format_ok"])
    os.makedirs(os.path.dirname(args.review) or ".", exist_ok=True)
    with open(args.review, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "answer", "格式合格", "数字核对通过", "备注"])
        for r in results:
            w.writerow([r["query"], r["output"], "是" if r["_format_ok"] else "否", "", ""])

    print(f"生成完成：{len(results)} 条，四段格式合格 {ok_n} 条（{ok_n/max(1,len(results))*100:.0f}%）")
    print(f"候选：{args.out}")
    print(f"人工审阅表：{args.review}")
    print()
    print("下一步（这一步不能省）：")
    print("  1. 打开审阅表，逐条核对 answer 里的每个数字，是否都能在 query 的素材里找到")
    print("  2. 找到的留，编出来的改掉或删掉整行")
    print("  3. 「格式合格=否」的行要补齐四段")
    print("  4. 「数字核对通过」填「是」的行另存为 data/raw/approved.jsonl")
    print("  5. 跑 python scripts/build_dataset.py --strict 生成正式训练集")


if __name__ == "__main__":
    main()
