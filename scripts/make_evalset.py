#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_evalset.py —— 造评测集 + 做泄漏体检

两个子命令：
  template  生成 200 条评测集的空白模板（带 2 行示例，照格式填）
  check     检查评测集与训练集是否重合（重合会让评测结果虚高）

评测集必须满足两条，否则整个评测没有意义：
  1) 不与训练集重合（尤其题干不能改两个词就搬过来）
  2) 覆盖真实分布：不能全是简单问题，要有「素材缺信息」「多条件冲突」「用户情绪差」这类难例

用法：
  python scripts/make_evalset.py template --n 200
  python scripts/make_evalset.py check --evalset data/eval/evalset.jsonl --train data/train.json
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import load_jsonl, simhash64, hamming, section_check, SECTIONS  # noqa: E402

TEMPLATE_HEADER = ["query", "answer", "difficulty", "category"]

EXAMPLES = [
    {
        "query": "用户情况：来电问上月为什么多扣了 15 元。\n检索素材：套餐=畅享 59 元档；增值业务=视频彩铃 6 元；套外流量=9 元；无其他扣费。",
        "answer": "【开场】我帮您把上月账单拆开了，一共三笔：套餐 59 元、视频彩铃 6 元、套外流量 9 元，多出来的就是后两笔。\n【核心权益】彩铃这个如果不需要，可以随时退，退了之后不再按月扣。\n【办理引导】发短信 QXSP 到 10086 就能退，或者在 App「我的-已订业务」里找到它点退订。\n【限制说明】本月已扣的 6 元不退还；套外流量是按实际用量结算的，没法追溯调整。",
        "difficulty": "简单",
        "category": "账单解释",
    },
    {
        "query": "用户情况：用户很生气，说你们乱扣费，要投诉到工信部。\n检索素材：账单构成=基础套餐 59 元；套外流量=12 元；用户近 3 个月无其他增值业务；上月有 1 次系统重复扣费已自动退款 12 元。",
        "answer": "【开场】您先别急，我把这个月的账一笔一笔跟您对一下。基础套餐 59 元，套外流量 12 元，就这两项。\n【核心权益】另外要跟您说明一下：上月确实有一笔 12 元的重复扣费，是我们的系统问题，这笔已经自动退回到您账户了，您可以在账单里看到退款记录。\n【办理引导】退款记录在 App「我的-账单详情」里能查到，我也可以把明细发到您手机上，您对着看。\n【限制说明】退的钱只能抵扣通信费，不能提现。这次的问题我这边会做个记录，如果您对处理结果不满意，可以继续投诉，我们按流程配合。",
        "difficulty": "困难",
        "category": "投诉安抚",
    },
]


def cmd_template(args):
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TEMPLATE_HEADER)
        w.writeheader()
        for e in EXAMPLES:
            w.writerow(e)
        # 铺空行，方便直接往下填
        for _ in range(max(0, args.n - len(EXAMPLES))):
            w.writerow({k: "" for k in TEMPLATE_HEADER})
    print(f"模板已生成：{out}（{args.n} 行，含 {len(EXAMPLES)} 行示例）")
    print()
    print("填写要求：")
    print("  1. query 里必须同时写「用户情况」和「检索素材」，这是模型的全部输入")
    print("  2. answer 必须是完整四段：" + " ".join(SECTIONS))
    print("  3. difficulty 填 简单/中等/困难，建议比例 5:3:2")
    print("  4. category 自由填（账单/权益/退订/投诉/宽带…），后面按类别看指标才有意义")
    print("  5. 填完用 Excel 另存为 UTF-8 CSV，再跑一次 --check 体检")
    print()
    print("⚠️ 别从训练数据里复制题干改两个词。评测集一旦和训练集重合，你算出来的提升是假的，")
    print("   而面试官问「测试集怎么保证没泄漏」时你答不上来，比没做评测更糟。")


def load_evalset(path):
    if path.endswith(".csv"):
        rows = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("query") or "").strip() and (r.get("answer") or "").strip():
                    rows.append({"query": r["query"], "answer": r["answer"],
                                 "difficulty": r.get("difficulty", ""), "category": r.get("category", "")})
        return rows
    return load_jsonl(path)


def cmd_check(args):
    ev = load_evalset(args.evalset)
    tr = load_jsonl(args.train)
    if not ev:
        print("评测集是空的（或表头/列名不对）。")
        sys.exit(1)

    ev_h = [simhash64((e.get("query") or "") + "||" + (e.get("answer") or "")) for e in ev]
    tr_h = [simhash64((t.get("input") or "") + "||" + (t.get("output") or "")) for t in tr]

    leaks = 0
    for i, h in enumerate(ev_h):
        if any(hamming(h, th) <= args.dedup for th in tr_h):
            leaks += 1
            if leaks <= 5:
                print(f"  [重合] {ev[i].get('query','')[:40]}...")

    bad_fmt = [i for i, e in enumerate(ev) if not section_check(e.get("answer") or "")[0]]
    diffs = {}
    for e in ev:
        d = (e.get("difficulty") or "未填").strip() or "未填"
        diffs[d] = diffs.get(d, 0) + 1

    print()
    print(f"评测集条数：{len(ev)}")
    print(f"与训练集重合：{leaks} 条" + ("  ❌ 必须处理" if leaks else "  ✅"))
    print(f"答案四段不合规：{len(bad_fmt)} 条" + ("  ❌" if bad_fmt else "  ✅"))
    print(f"难度分布：{diffs}")
    if len(ev) < 100:
        print(f"⚠️ 条数 {len(ev)} 偏少。100 条以下时，1 条差异就是 1 个百分点，结论不稳。建议 150~200。")
    if diffs.get("困难", 0) == 0:
        print("⚠️ 没有「困难」样本。全是简单题的话，微调前后差异会小到看不出，评测失去意义。")
    print()
    print("✅ 体检通过，可以用于评测" if (not leaks and not bad_fmt) else "❌ 体检未通过，先修上面标红项")


def cmd_split(args):
    """从训练集切出一份留出集做评测集，并从训练集里移除（避免泄漏）。

    这是小数据场景最实用的做法：你只有几百条，不可能另写一份评测集，
    那就切一块出来「考」模型，剩下的「教」模型。两者必须不重叠。
    """
    import random

    if not os.path.exists(args.train):
        print(f"找不到 {args.train}，先跑：python scripts/build_dataset.py")
        sys.exit(1)

    rows = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("训练集是空的，先跑 build_dataset.py")
        sys.exit(1)

    n_total = len(rows)
    k = args.n
    # --n 是「上限」而不是「必须」。小数据阶段绝不会因为凑不够 100 条就罢工，
    # 否则数据从 50 条补到 300 条的整个过程中，这一步会一直失败。
    if k > int(n_total * 0.35):
        cap = max(1, int(n_total * 0.35))
        if k > n_total:
            print(f"[提示] 训练集只有 {n_total} 条，凑不出 {args.n} 条考卷，"
                  f"按 35% 上限切 {cap} 条")
        else:
            print(f"[提示] 考卷最多占 35%，{args.n} 条已自动下调为 {cap} 条")
        k = cap
    if k < 1:
        print(f"训练集只有 {n_total} 条，切不出考卷。先补数据。")
        sys.exit(1)

    rng = random.Random(args.seed)
    idx = list(range(n_total))
    rng.shuffle(idx)
    hold = set(idx[:k])

    ev, rest = [], []
    for i, r in enumerate(rows):
        q = r.get("input") or r.get("query") or ""
        a = r.get("output") or r.get("answer") or ""
        if i in hold:
            ev.append({"query": q, "answer": a,
                       "difficulty": r.get("difficulty", ""),
                       "category": r.get("category", ""),
                       "rule_id": r.get("rule_id", "")})
        else:
            rest.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not args.no_remove:
        if not os.path.exists(args.backup):
            with open(args.backup, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(args.train, "w", encoding="utf-8") as f:
            for r in rest:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"评测集（考卷）：{len(ev)} 条 → {args.out}")
    print(f"训练集（教材）：{len(rest)} 条（原 {n_total} 条，已移除留出部分）")
    if not args.no_remove:
        print(f"原始训练集备份：{args.backup}")
        print("⚠️ 之后别再跑 build_dataset.py，否则会把考卷写回教材，评测分数会虚高。")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p3 = sub.add_parser("split", help="从训练集切出留出集做评测集（小数据推荐）")
    p3.add_argument("--train", default="data/train.json")
    p3.add_argument("--n", type=int, default=100, help="切出多少条做评测集")
    p3.add_argument("--out", default="data/eval/evalset.jsonl")
    p3.add_argument("--backup", default="data/train_full.json")
    p3.add_argument("--seed", type=int, default=42, help="固定随机种子，保证可复现")
    p3.add_argument("--no-remove", action="store_true", help="不修改训练集（不推荐，会泄漏）")
    p3.set_defaults(func=cmd_split)

    p1 = sub.add_parser("template", help="生成评测集模板")
    p1.add_argument("--n", type=int, default=200)
    p1.add_argument("--out", default="data/eval/evalset_template.csv")
    p1.set_defaults(func=cmd_template)

    p2 = sub.add_parser("check", help="泄漏与格式体检")
    p2.add_argument("--evalset", default="data/eval/evalset.jsonl")
    p2.add_argument("--train", default="data/train.json")
    p2.add_argument("--dedup", type=int, default=3)
    p2.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
