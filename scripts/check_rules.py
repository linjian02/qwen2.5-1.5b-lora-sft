# -*- coding: utf-8 -*-
"""
事实溯源校验（Fact Grounding Check）
=====================================
解决的问题：训练数据里最致命的错误不是"话说得不好"，而是
"output 里出现了 input 素材里没有的数字"——模型会把这种"编数字"的行为学走，
上线后对着用户报假资费。

本脚本对每条样本做 5 项检查：

  C1 四段结构      output 必须且只能包含 【开场】【核心权益】【办理引导】【限制说明】
  C2 数字溯源      output 里每个阿拉伯数字，必须能在 input 素材里找到同一个值
  C3 指令码溯源    output 里的大写业务指令（KT5G / QXFSR / KT10…）必须来自 input
  C4 禁词          不出现「亲」「宝子」「尊敬的客户」等模板腔
  C5 长度          每段 15~160 字，全文 120~600 字

用法：
    python scripts/check_rules.py                      # 校验 data/seed 下两份 jsonl
    python scripts/check_rules.py --file a.jsonl       # 校验指定文件
    python scripts/check_rules.py --allow 450,800      # 允许推算得出的数字（人工确认后）

退出码：0 全通过 / 1 有 FAIL
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = ROOT / "data" / "seed"


def default_files():
    """data/seed 下所有 jsonl 全查，新增数据文件不用改代码。"""
    got = sorted(SEED_DIR.glob("*.jsonl"))
    if not got:
        print(f"[!] {SEED_DIR} 下没有 jsonl 文件")
    return got

SEGMENTS = ["【开场】", "【核心权益】", "【办理引导】", "【限制说明】"]
# 结构性号码：公众服务号，出现在任何一条话术里都不算幻觉（用户可自行拨打）
STRUCTURAL = {float(x) for x in ["10086", "10010", "10000", "96110", "110", "119", "120"]}
BANNED = ["宝子", "亲亲", "尊敬的客户", "亲爱的用户", "么么哒", "嘻嘻", "小仙女"]
# 「亲」只在作为称呼时才算模板腔，避免误伤「父亲」「亲情」「亲属」
BANNED_RE = [(re.compile(r"(?:^|[\n，。！～~ ])亲[\s，。！～~]"), "亲（作称呼）")]

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# 业务指令码：2~6 位大写字母，可带数字，如 KT5G / QXFSR / KT10 / GMKT
CODE_RE = re.compile(r"\b[A-Z]{2,6}\d{0,2}\b")
# 时间写法：14:00-16:00，口语会说成「下午 2 点到 4 点」，这是换算不是幻觉
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


def time_variants(text: str) -> set:
    """24 小时制时间转成口语 12 小时制后可能出现的数值，允许这些值出现在 output"""
    allow = set()
    for h, m in TIME_RE.findall(text):
        h, m = int(h), int(m)
        for v in (h, h % 12, h % 12 + 12):
            if 0 <= v <= 24:
                allow.add(float(v))
        allow.add(float(m))
    return allow


def nums(text: str) -> set:
    """抽出文本中所有数字的数值形式（'23.6' -> 23.6, '40' -> 40.0）"""
    out = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(round(float(m), 4))
        except ValueError:
            pass
    return out


def fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


# 数值 + 单位：用来抓「素材有 40GB、答案写 40 元」这类同值不同单位的借用
UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(GB|MB|Mbps|分钟|小时|个月|元|块|天|年|次|期|条|秒|分|%|月|日)")
UNIT_CANON = {"块": "元", "月": "个月", "日": "天", "分": "分钟"}
# 网速的口语简写：320Mbps 说成 320M/320MB 属正常表达，不算借用
SPEED_ALIAS = {"Mbps", "MB", "M"}


def num_units(text: str) -> set:
    """抽出 (数值, 规范化单位) 集合"""
    out = set()
    for val, unit in UNIT_RE.findall(text):
        try:
            v = round(float(val), 4)
        except ValueError:
            continue
        out.add((v, UNIT_CANON.get(unit, unit)))
    return out


def context(text: str, value: float, width: int = 18) -> str:
    """取数字在原文中的上下文，便于人工判断"""
    target = fmt(value)
    idx = text.find(target)
    if idx < 0:
        idx = text.find(target.rstrip("0").rstrip("."))
    if idx < 0:
        return text[:40]
    left = max(0, idx - width)
    right = min(len(text), idx + len(target) + width)
    return ("…" if left else "") + text[left:right].replace("\n", " ") + ("…" if right < len(text) else "")


def check_one(row: dict, allow: set):
    """返回 (fail_list, warn_list)"""
    inp = row.get("input", "")
    out = row.get("output", "")
    fails, warns = [], []

    # C1 四段结构
    if not out.strip():
        fails.append("C1 output 为空")
        return fails, warns
    positions = []
    for seg in SEGMENTS:
        cnt = out.count(seg)
        if cnt == 0:
            fails.append(f"C1 缺少段落 {seg}")
        elif cnt > 1:
            fails.append(f"C1 段落 {seg} 重复 {cnt} 次")
        else:
            positions.append(out.find(seg))
    if len(positions) == len(SEGMENTS) and positions != sorted(positions):
        fails.append("C1 四段顺序错乱")
    # output 里不许出现第五个【xx】
    extra = set(re.findall(r"【[^】]{1,8}】", out)) - set(SEGMENTS)
    if extra:
        fails.append(f"C1 出现多余段落标记 {sorted(extra)}")

    # C2 数字溯源：output 的每个数字必须来自 input、公共服务号、时间换算，或人工白名单
    src = nums(inp) | STRUCTURAL | allow | time_variants(inp)
    bad = sorted(nums(out) - src)
    if bad:
        detail = "；".join(f"{fmt(v)} → {context(out, v)}" for v in bad)
        fails.append(f"C2 数字无出处 [{detail}]")

    # C3 指令码溯源
    src_codes = set(CODE_RE.findall(inp))
    out_codes = set(CODE_RE.findall(out))
    # 允许 output 复述 input 里出现的指令；新增即失败
    novel_codes = {c for c in out_codes - src_codes if not c.isdigit()}
    if novel_codes:
        fails.append(f"C3 指令码无出处 {sorted(novel_codes)}")

    # C4 禁词
    hit = [w for w in BANNED if w in out]
    hit += [label for pat, label in BANNED_RE if pat.search(out)]
    if hit:
        fails.append(f"C4 出现模板腔禁词 {hit}")
    if out.count("！") > 2:
        warns.append(f"C4 感叹号 {out.count('！')} 个，偏多")

    # C6 同值不同单位：素材有 40GB、答案写 40 元，数值集合比对会放过，这里单独提醒。
    # 这是纯数字追溯的天然盲区（也是 51 条数据阶段漏掉「每月省 40 元」的原因）。
    in_units = num_units(inp)
    out_units = num_units(out)
    in_vals = {v for v, _ in in_units}
    odd = sorted({(v, u) for v, u in out_units
                  if v in in_vals and (v, u) not in in_units
                  and not (u in SPEED_ALIAS and any(iu in SPEED_ALIAS for iv, iu in in_units if iv == v))})
    if odd:
        detail = "；".join(f"{fmt(v)} {u}" for v, u in odd)
        warns.append(f"C6 同值不同单位，可能借用素材里的数字 [{detail}]")

    # C5 长度
    body = out
    for seg in SEGMENTS:
        body = body.replace(seg, "|")
    parts = [p.strip() for p in body.split("|") if p.strip()]
    if len(parts) == 4:
        for seg, seg_text in zip(SEGMENTS, parts):
            n = len(seg_text)
            if n < 15:
                fails.append(f"C5 {seg} 过短（{n} 字），信息职责没交代清楚")
            elif n > 160:
                warns.append(f"C5 {seg} 过长（{n} 字），读起来像公告")
    total = len(out.replace("\n", ""))
    if total < 120:
        fails.append(f"C5 全文 {total} 字，过短")
    elif total > 600:
        warns.append(f"C5 全文 {total} 字，偏长")

    return fails, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=None, help="待校验 jsonl，可多次指定")
    ap.add_argument("--allow", default="", help="允许的推算数字，逗号分隔，如 450,800")
    args = ap.parse_args()

    allow = set()
    for x in args.allow.split(","):
        x = x.strip()
        if x:
            try:
                allow.add(round(float(x), 4))
            except ValueError:
                pass

    files = [Path(f) for f in args.file] if args.file else default_files()
    report_lines = ["# 事实溯源校验报告", ""]
    total_n = total_fail = total_warn = 0

    for fp in files:
        if not fp.exists():
            print(f"[跳过] {fp} 不存在")
            continue
        rows = [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
        n_fail = n_warn = 0
        detail = []
        for i, row in enumerate(rows, 1):
            f, w = check_one(row, allow)
            total_n += 1
            if f:
                n_fail += 1
                detail.append(f"\n### 第 {i} 条 ✗\n" + "\n".join(f"- {x}" for x in f))
            if w:
                n_warn += 1
                detail.append(f"\n### 第 {i} 条 ⚠\n" + "\n".join(f"- {x}" for x in w))
        total_fail += n_fail
        total_warn += n_warn
        head = f"## {fp.name} — {len(rows)} 条，FAIL {n_fail}，WARN {n_warn}"
        print(f"\n{head}")
        print("=" * len(head))
        for d in detail:
            print(d)
        report_lines.append(head)
        report_lines.append("")
        report_lines.extend(detail if detail else ["全部通过。"])
        report_lines.append("")

    verdict = "PASS" if total_fail == 0 else "FAIL"
    summary = f"\n总计 {total_n} 条：FAIL {total_fail}，WARN {total_warn} → {verdict}"
    print(summary)
    report_lines.insert(2, f"总计 {total_n} 条：FAIL {total_fail}，WARN {total_warn} → **{verdict}**\n")

    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "fact_check_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告已写入 out/fact_check_report.md")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
