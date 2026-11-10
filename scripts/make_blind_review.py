#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_blind_review.py —— 生成人工盲评表，再汇总胜率

为什么要做：自动指标只能测结构（有没有四段、数字对不对），测不出「读起来像不像人说的话」。
算法岗面试官一定会问「你怎么判断生成质量」，只答「我看了几条觉得变好了」是不够的 ——
盲评 + 胜率是标准答案，成本也就一两个小时。

两个子命令：
  make     生成盲评 CSV（A/B 顺序随机，不告诉你哪个是微调后的）
  summary  读回填好的 CSV，算胜率

用法：
  python scripts/make_blind_review.py make --a out/pred_base.jsonl --b out/pred_sft.jsonl --n 60
  # 用 Excel 打开 out/blind_review.csv，在 winner 列填 A / B / 平
  python scripts/make_blind_review.py summary --in out/blind_review.csv
"""

import argparse
import csv
import json
import os
import random
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_make(args):
    a = load(args.a)
    b = load(args.b)
    n = min(args.n, len(a), len(b))
    if n < len(a):
        print(f"从 {len(a)} 条中随机取 {n} 条做盲评（全量盲评太耗时间，60 条足够看出趋势）")

    rnd = random.Random(args.seed)
    idx = rnd.sample(range(min(len(a), len(b))), n)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    key = []
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["序号", "用户情况与素材", "回复A", "回复B", "哪个更好(A/B/平)", "备注"])
        for k, i in enumerate(idx, 1):
            ra, rb = a[i]["pred"], b[i]["pred"]
            if rnd.random() < 0.5:
                left, right, mapping = ra, rb, {"A": "a", "B": "b"}
            else:
                left, right, mapping = rb, ra, {"A": "b", "B": "a"}
            key.append({"序号": k, **mapping})
            w.writerow([k, a[i]["query"], left, right, "", ""])

    with open(args.out.replace(".csv", "_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)

    print(f"盲评表已生成：{args.out}（{n} 条，A/B 顺序已随机）")
    print(f"答案对照表（别提前看）：{args.out.replace('.csv', '_key.json')}")
    print()
    print("评判标准，按顺序比：")
    print("  1. 有没有编造素材里没有的信息（编了就一定判它输）")
    print("  2. 关键条件说全了没有（门槛、时效、限制）")
    print("  3. 读起来像不像真人说话（不是模板堆砌）")
    print("填完 winner 列保存，再跑 summary。")


def cmd_summary(args):
    path = args.inp
    keypath = path.replace(".csv", "_key.json")
    if not os.path.exists(keypath):
        print(f"找不到对照表 {keypath}")
        sys.exit(1)
    with open(keypath, encoding="utf-8") as f:
        key = {int(k["序号"]): k for k in json.load(f)}

    win = {"a": 0, "b": 0, "tie": 0, "skip": 0}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("哪个更好(A/B/平)") or "").strip()
            try:
                k = int(r.get("序号") or 0)
            except ValueError:
                continue
            if k not in key or v == "":
                win["skip"] += 1
                continue
            if v in ("平", "tie", "T"):
                win["tie"] += 1
                continue
            side = key[k].get(v.upper())
            if side in ("a", "b"):
                win[side] += 1
            else:
                win["skip"] += 1

    total = win["a"] + win["b"] + win["tie"]
    if total == 0:
        print("没有有效评分。检查 winner 列是否填了 A/B/平。")
        sys.exit(1)

    a_rate = win["a"] / total * 100
    b_rate = win["b"] / total * 100
    # 排除平局后的胜率，这个数更能区分优劣
    decided = win["a"] + win["b"]
    a_dec = win["a"] / decided * 100 if decided else 0

    print(f"有效评分：{total} 条（跳过 {win['skip']} 条）")
    print(f"  微调胜（A）：{win['a']} 次")
    print(f"  基座胜（B）：{win['b']} 次")
    print(f"  平局：{win['tie']} 次")
    print()
    print(f"微调模型胜率：{a_rate:.1f}%（含平局） / {a_dec:.1f}%（不含平局）")
    print()
    if total < 30:
        print("⚠️ 样本少于 30 条，胜率波动会很大，建议补到 60 条以上再写进简历。")
    print(f"简历/README 可以写：「{total} 条盲评中微调模型胜率 {a_rate:.0f}%」")
    print("面试追问「怎么保证公平」时的回答要点：A/B 顺序随机、评分人不知道哪个是微调后的、"
          "共 X 条、判定标准事先定好（先看有没有编造，再看条件完整性，最后看语感）。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make")
    m.add_argument("--a", default="out/pred_base.jsonl")
    m.add_argument("--b", default="out/pred_sft.jsonl")
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--out", default="out/blind_review.csv")
    m.set_defaults(func=cmd_make)

    s = sub.add_parser("summary")
    s.add_argument("--in", dest="inp", default="out/blind_review.csv")
    s.set_defaults(func=cmd_summary)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
