#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score.py —— 从预测结果算指标，输出可直接贴进简历/README 的对比表

四个指标，各自回答一个面试必问的问题：
  格式合规率  → 「输出结构稳不稳」四段是否齐全且顺序正确
  数字保留率  → 「该说的说了吗」素材里的关键数字，输出覆盖了多少（漏 = 信息丢失）
  数字幻觉率  → 「不该说的说了吗」输出出现了素材里没有的数字（编 = 幻觉）
  延迟 P50/P99 → 「能不能上线」

另外把人工标注的标准答案（gold）也按同样口径算一遍，作为「天花板」参照。
如果 gold 的格式合规率不是 100%，说明你的评测脚本有 bug，不是模型的问题。

用法：
  python scripts/score.py --pred out/pred_base.jsonl out/pred_sft.jsonl --out out/score_report.md
"""

import argparse
import json
import os
import re
import statistics

SECTIONS = ["【开场】", "【核心权益】", "【办理引导】", "【限制说明】"]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 顺序词噪音：模型常写「第 1 点」「3 个渠道」，这类小整数不计入幻觉
NOISE_INTS = {str(i) for i in range(0, 11)}

# 主动声明信息缺失：判断「难例训练有没有教会模型说不知道」
HEDGE = ["资料里没有", "资料里没写", "资料没写", "资料没给", "资料里没给", "没有这笔",
         "没给这个", "查不到", "不敢", "不确定", "没法确认", "不能确定", "没有具体",
         "不给您编", "不报数", "以页面为准", "以账单页为准"]

# 公共服务号：模型在「您打 10086 查一下」这类引导里提到，不算编造资费
PUBLIC_NUMS = {"10086", "10010", "10000", "96110", "110", "119", "120"}


def section_ok(text):
    if not text:
        return False
    pos = []
    for tag in SECTIONS:
        i = text.find(tag)
        if i < 0:
            return False
        pos.append(i)
    return all(pos[i] < pos[i + 1] for i in range(len(pos) - 1))


def nums(text):
    return set(NUM_RE.findall(text or ""))


def pct(a, b):
    return 0.0 if b == 0 else a / b * 100


def pctl(vals, q):
    if not vals:
        return 0.0
    v = sorted(vals)
    i = min(len(v) - 1, int(round(q * (len(v) - 1))))
    return v[i]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(rows):
    n = len(rows)
    fmt = sum(1 for r in rows if section_ok(r.get("pred")))

    # 数字保留：素材里的数字被输出覆盖的比例
    keep_hit = keep_all = 0
    # 数字幻觉：输出里素材没有的数字比例
    hall_bad = hall_all = 0
    lens, lats = [], []

    for r in rows:
        qn = nums(r.get("query"))
        pn = nums(r.get("pred"))
        keep_all += len(qn)
        keep_hit += len(qn & pn)
        extra = {x for x in (pn - qn) if x not in NOISE_INTS and x not in PUBLIC_NUMS}
        hall_all += len(pn)
        hall_bad += len(extra)
        if r.get("pred"):
            lens.append(len(r["pred"]))
        if r.get("latency_ms"):
            lats.append(r["latency_ms"])

    return {
        "n": n,
        "fmt": pct(fmt, n),
        "keep": pct(keep_hit, keep_all),
        "hall": pct(hall_bad, hall_all),
        "len_avg": int(round(statistics.mean(lens))) if lens else 0,
        "p50": int(round(pctl(lats, 0.50))) if lats else None,
        "p99": int(round(pctl(lats, 0.99))) if lats else None,
    }


def hedge_rate(rows):
    """主动声明信息缺失的比例。

    只统计有预测的样本。这个数比总体幻觉率更能说明「难例训练」有没有用：
    难例题上应该明显高于普通题，否则模型还是在硬编。
    """
    sub = [r for r in rows if r.get("pred")]
    if not sub:
        return None
    hit = sum(1 for r in sub if any(h in r["pred"] for h in HEDGE))
    return pct(hit, len(sub))


def hall_rate(rows):
    sub = [r for r in rows if r.get("pred")]
    if not sub:
        return None
    return evaluate(sub).get("hall")


def evaluate_gold(rows):
    """把人工答案当预测跑一遍同样口径，作为天花板"""
    fake = [{"query": r.get("query"), "pred": r.get("gold"), "difficulty": r.get("difficulty", "")}
            for r in rows if r.get("gold")]
    return (fake, evaluate(fake)) if fake else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="一个或多个 pred_*.jsonl")
    ap.add_argument("--out", default="out/score_report.md")
    args = ap.parse_args()

    runs = {}
    for p in args.pred:
        rows = load(p)
        tag = rows[0].get("tag", os.path.basename(p)) if rows else os.path.basename(p)
        runs[tag] = (rows, evaluate(rows))
        # 用第一次出现的 run 做 gold 参照即可
        if "gold" not in runs:
            g = evaluate_gold(rows)
            if g:
                runs["gold"] = g

    tags = [t for t in runs if t != "gold"]
    order = ["base"] + [t for t in tags if t != "base"]
    order = [t for t in order if t in runs]
    cols = order + (["gold"] if "gold" in runs else [])

    def row(label, key, fmt="{:.1f}%"):
        cells = []
        for c in cols:
            v = runs[c][1].get(key)
            cells.append("—" if v is None else (fmt.format(v) if isinstance(v, float) else str(v)))
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        "# 微调评测报告", "",
        f"评测集：{runs[cols[0]][1]['n']} 条（gold 列为人工答案，代表上限参照）", "",
        "| 指标 | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
        row("格式合规率 ↑", "fmt"),
        row("关键数字保留率 ↑", "keep"),
        row("数字幻觉率 ↓", "hall"),
        row("平均输出字数", "len_avg", "{}"),
        row("P50 延迟 (ms)", "p50", "{}"),
        row("P99 延迟 (ms)", "p99", "{}"),
    ]

    # 按难度分组：只看总分会掩盖「简单题都会、难题全崩」
    if runs[cols[0]][0]:
        lines += ["", "## 按难度分组（格式合规率）", "",
                  "| 难度 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for d in ["简单", "中等", "困难"]:
            cells = []
            for c in cols:
                rows = [r for r in runs[c][0] if (r.get("difficulty") or "") == d]
                cells.append(f"{pct(sum(1 for r in rows if section_ok(r.get('pred'))), len(rows)):.1f}%" if rows else "—")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")

        # 难例到底有没有用，看这两项的分组差异，而不是看总体幻觉率
        levels = []
        for r in runs[cols[0]][0]:
            lv = r.get("difficulty") or "未填"
            if lv not in levels:
                levels.append(lv)
        lines += ["", "## 按难度分组（幻觉率 / 主动声明缺失率）", "",
                  "难例题上「主动声明缺失」应明显高于普通题；若两者接近，说明难例比例还不够。", "",
                  "| 难度 | 条数 | " + " | ".join(f"{c} 幻觉率 ↓" for c in cols)
                  + " | " + " | ".join(f"{c} 声明缺失 ↑" for c in cols) + " |",
                  "|---|---|" + "---|" * (len(cols) * 2)]
        for lv in levels:
            sub = [r for r in runs[cols[0]][0] if (r.get("difficulty") or "未填") == lv]
            cells = []
            for key in ("hall", "hedge"):
                for c in cols:
                    rows = [r for r in runs[c][0] if (r.get("difficulty") or "未填") == lv]
                    v = hall_rate(rows) if key == "hall" else hedge_rate(rows)
                    cells.append("—" if v is None else f"{v:.1f}%")
            lines.append(f"| {lv} | {len(sub)} | " + " | ".join(cells) + " |")

    lines += ["", "## 怎么读这张表", ""]
    lines += [
        "- **格式合规率**是微调最直接的效果。base 模型通常只有 10~40%，因为它不知道你要四段；"
        "SFT 之后应到 95% 以上。这一项不动，说明训练没生效（先查 template 是否写对）。",
        "- **数字保留率**低 = 模型把关键条件丢了，线上就是「说了但不准」。"
        "低于 90% 要回去查训练数据里是不是把数字截断或改写了。",
        "- **数字幻觉率**是最该盯的一项。素材里没有的数字被编出来，在权益/资费场景是事故。"
        "SFT 能压到 5% 以下说明模型学到了「不编」；压不下去就加大「素材中无相关信息时明确说不知道」的样本比例。",
        "- **P99 延迟**要用同一台机器、同一 batch 设置测，跨环境对比没有意义。",
        "- 如果 base 和 sft 指标几乎一样，先怀疑三件事：adapter 没加载、评测集撞了训练集、"
        "或者训练样本太少（<100 条）模型根本没学进去。",
    ]

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n已写入：{args.out}")


if __name__ == "__main__":
    main()
