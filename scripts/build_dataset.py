#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset.py —— 把零散的话术素材整理成 LLaMA-Factory 可直接训练的 Alpaca 格式

零第三方依赖，只用 Python 标准库（避免在服务器上装包踩坑）。

支持输入（放在 data/raw/ 下，可混合）：
  1) *.jsonl  每行一个 JSON，字段支持两种写法：
       {"instruction": ..., "input": ..., "output": ...}   ← Alpaca 标准
       {"query": ..., "answer": ..., "product": ...}       ← 简写，脚本自动转
  2) *.csv    表头需含 query/answer（或 问题/回答），其余列忽略

做四件事：
  ① 清洗：空白归一、去 markdown 残留、去口水前缀、长度过滤
  ② 校验：检查 output 是否含「【开场】【核心权益】【办理引导】【限制说明】」四段，
          缺段的数据单独挑出来（这是评测指标「格式合规率」的地基）
  ③ 去重：64 位 SimHash + 4 段分桶（汉明距离 ≤ 阈值视为重复），O(n) 级别
  ④ 切分：输出 data/train.json，并可选与评测集做交叉去重检查

用法：
  python scripts/build_dataset.py                       # 默认读 data/raw
  python scripts/build_dataset.py --strict             # 缺段数据直接剔除
  python scripts/build_dataset.py --check-evalset data/eval/evalset.jsonl
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

# ---------- 四段式标签（与评测脚本共用，改这里就够） ----------
SECTIONS = ["【开场】", "【核心权益】", "【办理引导】", "【限制说明】"]

# 禁止出现的口水词：训练数据里出现，模型会学过去
BANNED_WORDS = ["亲~", "亲，", "宝子", "家人们", "么么哒", "给您比心"]

# 全角/半角与空白归一
WS_RE = re.compile(r"[ \t\u3000]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
MD_RE = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s*)", re.M)


# ============================ 清洗 ============================
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WS_RE.sub(" ", s)
    s = MULTI_NL_RE.sub("\n\n", s)
    s = MD_RE.sub("", s)
    for w in BANNED_WORDS:
        s = s.replace(w, "")
    # 去掉行首行尾空白，但保留段内换行
    lines = [ln.strip() for ln in s.split("\n")]
    s = "\n".join(ln for ln in lines if ln != "")
    return s.strip()


def section_check(text):
    """返回 (是否合规, 缺失的段, 顺序是否正确)"""
    pos = []
    for tag in SECTIONS:
        i = text.find(tag)
        if i < 0:
            pos.append(-1)
        else:
            pos.append(i)
    missing = [SECTIONS[i] for i, p in enumerate(pos) if p < 0]
    if missing:
        return False, missing, False
    in_order = all(pos[i] < pos[i + 1] for i in range(len(pos) - 1))
    return True, [], in_order


# ============================ SimHash ============================
def _tokens(text):
    """中文按字 bigram，英文/数字按整词。短文本兜底用单字。"""
    t = re.sub(r"\s+", "", text)
    grams = set()
    for m in re.finditer(r"[A-Za-z0-9]+", t):
        grams.add(m.group(0).lower())
    han = re.sub(r"[A-Za-z0-9]+", "", t)
    if len(han) >= 2:
        grams.update(han[i:i + 2] for i in range(len(han) - 1))
    elif han:
        grams.add(han)
    return grams or {t}


def simhash64(text):
    v = [0] * 64
    for g in _tokens(text):
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


class SimHashIndex:
    """4 段分桶：汉明距离 ≤3 时必然至少有一段完全相同，所以只需比对同桶候选。"""

    def __init__(self, bands=4):
        self.bands = bands
        self.width = 64 // bands
        self.mask = (1 << self.width) - 1
        self.buckets = defaultdict(list)
        self.items = []  # [(hash, payload)]

    def _keys(self, h):
        return [(i, (h >> (i * self.width)) & self.mask) for i in range(self.bands)]

    def add(self, h, payload):
        idx = len(self.items)
        self.items.append((h, payload))
        for k in self._keys(h):
            self.buckets[k].append(idx)

    def find_dup(self, h, threshold):
        cand = set()
        for k in self._keys(h):
            cand.update(self.buckets.get(k, ()))
        for idx in cand:
            if hamming(h, self.items[idx][0]) <= threshold:
                return self.items[idx][1]
        return None


# ============================ 读取 ============================
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [跳过] {os.path.basename(path)} 第 {ln} 行 JSON 解析失败：{e}")
    return rows


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            q = r.get("query") or r.get("问题") or r.get("question") or ""
            a = r.get("answer") or r.get("回答") or r.get("output") or ""
            rows.append({"query": q, "answer": a, "product": r.get("product", "")})
    return rows


DEFAULT_INSTR = ("你是运营商权益营销话术助手。请根据【检索素材】生成一段可直接发给用户的话术。"
                 "要求：严格按【开场】【核心权益】【办理引导】【限制说明】四段输出；"
                 "只能使用素材中出现过的数字与条件，素材里没有的信息一律不许补；"
                 "若素材不足以回答用户，就在对应段落明确说明缺哪项、并给出下一步，不要猜测；"
                 "语气口语化，不使用「亲」「宝子」等词。")

# 训练用不到、但评测要用的元字段，必须透传到 train.json
KEEP_META = ("difficulty", "category", "source", "rule_id")


def normalize_row(r):
    """统一成 Alpaca 三字段，并保留评测所需的元字段"""
    meta = {k: r[k] for k in KEEP_META if r.get(k)}
    if "output" in r and "instruction" in r:
        return {"instruction": r["instruction"], "input": r.get("input", ""),
                "output": r["output"], **meta}
    q = r.get("query") or r.get("question") or ""
    a = r.get("answer") or r.get("output") or ""
    product = r.get("product") or r.get("产品") or ""
    instr = DEFAULT_INSTR
    if product:
        instr += f"当前产品线：{product}。"
    return {"instruction": instr, "input": q, "output": a, **meta}


# ============================ 主流程 ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw", help="原始素材目录")
    ap.add_argument("--seed", default="data/seed", help="种子目录（读其中所有 jsonl）或单个文件")
    ap.add_argument("--out", default="data/train.json")
    ap.add_argument("--report", default="out/dataset_report.md")
    ap.add_argument("--min-len", type=int, default=60, help="output 最短字数")
    ap.add_argument("--max-len", type=int, default=1200, help="output 最长字数")
    ap.add_argument("--dedup", type=int, default=3, help="SimHash 汉明距离阈值，≤ 视为重复")
    ap.add_argument("--strict", action="store_true", help="四段不全的数据直接剔除（推荐）")
    ap.add_argument("--check-evalset", default="", help="可选：与评测集做交叉去重检查")
    args = ap.parse_args()

    files = []
    if args.seed:
        if os.path.isdir(args.seed):
            for fn in sorted(os.listdir(args.seed)):
                if fn.lower().endswith(".jsonl"):
                    files.append(os.path.join(args.seed, fn))
        elif os.path.exists(args.seed):
            files.append(args.seed)
    if os.path.isdir(args.src):
        for fn in sorted(os.listdir(args.src)):
            if fn.lower().endswith((".jsonl", ".json", ".csv")):
                files.append(os.path.join(args.src, fn))

    if not files:
        print("没有找到任何素材。请把 jsonl/csv 放进 data/raw/，或保留 data/seed/ 下的种子数据。")
        sys.exit(1)

    raw = []
    for p in files:
        if p.endswith(".csv"):
            rs = load_csv(p)
        else:
            rs = load_jsonl(p)
        raw.extend(rs)
        print(f"读取 {p}：{len(rs)} 条")

    total = len(raw)
    kept, dropped = [], Counter()
    seen_hash = SimHashIndex()

    for r in raw:
        rec = normalize_row(r)
        q = clean_text(rec.get("input", ""))
        a = clean_text(rec.get("output", ""))
        instr = clean_text(rec.get("instruction", ""))

        if not q or not a:
            dropped["字段为空"] += 1
            continue
        if len(a) < args.min_len:
            dropped["output 过短"] += 1
            continue
        if len(a) > args.max_len:
            dropped["output 过长"] += 1
            continue

        ok, missing, in_order = section_check(a)
        if not ok:
            dropped["四段不全：" + "/".join(missing)] += 1
            if args.strict:
                continue
        elif not in_order:
            dropped["四段顺序错乱"] += 1
            if args.strict:
                continue

        h = simhash64(q + "||" + a)
        dup = seen_hash.find_dup(h, args.dedup)
        if dup is not None:
            dropped["近重复"] += 1
            continue
        seen_hash.add(h, len(kept))

        # 额外字段要带到 train.json，否则 make_evalset split 时 difficulty 丢失，
        # 评测报告就无法按「简单/中等/困难」分组，看不出难例上的提升。
        item = {"instruction": instr, "input": q, "output": a, "_hash": h}
        for extra in ("difficulty", "category", "source", "rule_id"):
            if rec.get(extra):
                item[extra] = rec[extra]
        kept.append(item)

    # 交叉去重：训练集不能污染评测集
    leak = 0
    if args.check_evalset and os.path.exists(args.check_evalset):
        ev = load_jsonl(args.check_evalset)
        ev_hash = [simhash64((e.get("query") or e.get("input") or "") + "||" + (e.get("answer") or e.get("output") or ""))
                   for e in ev]
        before = len(kept)
        kept = [r for r in kept
                if not any(hamming(r["_hash"], eh) <= args.dedup for eh in ev_hash)]
        leak = before - len(kept)
        if leak:
            dropped["与评测集重合（已剔除）"] += leak

    for r in kept:
        r.pop("_hash", None)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = [len(r["output"]) for r in kept] or [0]
    lens.sort()
    p = lambda q: lens[min(len(lens) - 1, int(len(lens) * q))]
    compliant = sum(1 for r in kept if section_check(r["output"])[0])

    lines = [
        "# 训练数据体检报告", "",
        f"- 输入文件：{len(files)} 个",
        f"- 原始条数：**{total}**",
        f"- 入库条数：**{len(kept)}**（保留率 {len(kept)/max(1,total)*100:.1f}%）",
        f"- 四段格式合规：**{compliant}/{len(kept)}**（{compliant/max(1,len(kept))*100:.1f}%）",
        f"- output 字数：最短 {lens[0]}，P50 {p(0.5)}，P90 {p(0.9)}，最长 {lens[-1]}",
        "",
        "## 剔除明细", "",
    ]
    if dropped:
        for k, v in dropped.most_common():
            lines.append(f"- {k}：{v}")
    else:
        lines.append("- 无")
    if leak:
        lines.append("")
        lines.append(f"> ⚠️ 有 {leak} 条与评测集高度重合，已自动剔除。若这个数很大，说明评测集是从训练数据里抽的，"
                     "评测结果会虚高，需要手工重写评测集。")

    # 自查清单
    lines += ["", "## 达标线自查", ""]
    n = len(kept)
    checks = [
        (n >= 300, f"条数 ≥ 300（当前 {n}）", "条数太少，模型只会背不会泛化；300 是能看出趋势的下限"),
        (compliant / max(1, n) >= 0.95, f"四段合规率 ≥ 95%（当前 {compliant/max(1,n)*100:.1f}%）",
         "训练数据格式不一，评测时格式合规率会直接垮"),
        (p(0.5) >= 80, f"中位长度 ≥ 80 字（当前 {p(0.5)}）", "太短的话学不到「限制说明」这种长句表达"),
    ]
    for ok, desc, hint in checks:
        lines.append(f"- {'✅' if ok else '❌'} {desc}" + ("" if ok else f" —— {hint}"))

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("\n".join(lines))
    print(f"\n已写出：{args.out}")
    print(f"报告：{args.report}")


if __name__ == "__main__":
    main()
