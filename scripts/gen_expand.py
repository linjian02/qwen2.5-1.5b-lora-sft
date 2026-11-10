# -*- coding: utf-8 -*-
"""扩量生成器：R01~R33 场景包 + 人格改写。

设计原则（与规则表第三节一致）：
  1. 每条样本的 input 素材与 output 答案来自**同一份字段**，数字天然可溯源；
  2. 需要结论值的（折合金额/违约金/节省额）一律把结论写进素材，不让模型算；
  3. 难例统一为「素材缺某字段」→ 答案必须明说缺哪项 + 给下一步。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from scen_pack_a import SCEN_A  # noqa: E402
from scen_pack_b import SCEN_B  # noqa: E402
from scen_pack_c import SCEN_C  # noqa: E402

SCEN = SCEN_A + SCEN_B + SCEN_C

INSTR = (
    "你是运营商权益营销话术助手。请根据【检索素材】生成一段可直接发给用户的话术。"
    "要求：严格按【开场】【核心权益】【办理引导】【限制说明】四段输出；"
    "只能使用素材中出现过的数字与条件，素材里没有的信息一律不许补；"
    "若素材不足以回答用户，就在对应段落明确说明缺哪项、并给出下一步，不要猜测；"
    "语气口语化，不使用「亲」「宝子」等词。"
)

SEGS = ["【开场】", "【核心权益】", "【办理引导】", "【限制说明】"]

# 人格改写：只换问法与开场口吻，事实字段不动
PERSONA = [
    ("长者", "，我这个年纪不太懂这些，你慢慢说", "您别急，我一条一条给您说清楚。"),
    ("着急", "，挺急的，麻烦你直接说重点", "您先别急，能查到的我一次给您说完。"),
    ("怀疑", "，我怀疑你们是不是乱收费，得给我讲明白", "您有这个疑问很正常，我把规则原样讲一遍，您自己判断。"),
]


def make_row(rid, cat, q, facts, ans, diff):
    a = ans.strip()
    for s in SEGS:
        if a.count(s) != 1:
            raise SystemExit(f"{rid} 段落结构异常：{s} 出现 {a.count(s)} 次")
    if not a.startswith("【开场】"):
        raise SystemExit(f"{rid} 未以【开场】开头")
    return {
        "instruction": INSTR,
        "input": f"用户情况：{q}\n检索素材：{facts}",
        "output": a,
        "difficulty": diff,
        "category": cat,
        "rule_id": rid,
    }


def persona_rewrite(row, suffix, lead):
    """保留事实，只改问法前缀与开场首句口吻"""
    a = row["output"]
    head, rest = a.split("【开场】", 1)
    first, tail = rest.split("【核心权益】", 1)
    new_open = lead + " " + first.strip()
    if len(new_open) > 155:
        new_open = lead + " " + first.strip()[:120].rstrip("，。") + "。"
    out = "【开场】" + new_open + "\n【核心权益】" + tail
    q = row["input"].split("\n", 1)[0].replace("用户情况：", "", 1) + suffix
    return {
        "instruction": row["instruction"],
        "input": f"用户情况：{q}\n" + row["input"].split("\n", 1)[1],
        "output": out,
        "difficulty": row["difficulty"],
        "category": row["category"],
        "rule_id": row["rule_id"],
    }


def main():
    rows = []
    for s in SCEN:
        base = make_row(s["rid"], s["cat"], s["q"], s["facts"], s["a"], "普通")
        rows.append(base)
        for _name, suffix, lead in PERSONA:
            rows.append(persona_rewrite(base, suffix, lead))
        for h in s["hards"]:
            rows.append(make_row(s["rid"], s["cat"], h["q"], h["facts"], h["a"], "困难"))

    out = ROOT / "data" / "seed" / "expand.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_hard = sum(1 for r in rows if r["difficulty"] == "困难")
    print(f"场景 {len(SCEN)} 个，生成 {len(rows)} 条（普通 {len(rows) - n_hard} / 困难 {n_hard}）")
    print(f"输出 {out}")


if __name__ == "__main__":
    main()
