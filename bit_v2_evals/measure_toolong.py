"""Token-budget breakdown of the v3 uniform bit CoTs (eden SLURM, CPU).
For every narrator-ok real: total completion tokens, and tokens attributable
to the Whole-word check section vs the per-bit body. Then SIMULATE four safe
trims (no per-bit derivation-logic change) and report how many of the 117
toolong each recovers:
  T0 current
  T1 drop the "Selected" block (re-lists the 8 Matched rules verbatim)
  T2 fully compact example-checks (ex0 also loses its term breakdown)
  T3 cap example-checks at 5 lines (+ "(+K more examples match)")
  T4 = T1+T2+T3 combined
Reports toolong count, p50/p90/max for each, plus the verdict-shape split of
whatever remains toolong under T4.
"""
import json, re, sys
B = '/mnt/evafs/groups/re-com/mgromadzki'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
sys.path.insert(0, 'src')
from transformers import AutoTokenizer
import reasoners.bit_manipulation_zyx as Z
from reasoners.store_types import Problem, Example

CHECK_RE = re.compile(r"^([01]{8}) (?:\S+=[01]{8} ?)+-> ([01]{8} vs [01]{8} (?:yes|no))$")
CAP = 7680


def transform(cot, drop_selected, compact_all, cap_ex):
    lines = cot.split("\n")
    out = []
    i = 0
    seen = 0
    in_check = False
    skipping_selected = False
    extra = 0
    while i < len(lines):
        ln = lines[i]
        if drop_selected and ln == "Selected":
            # skip "Selected" + the following 8 "i expr" lines + trailing blank
            j = i + 1
            while j < len(lines) and re.match(r"^\d ", lines[j]):
                j += 1
            if j < len(lines) and lines[j] == "":
                j += 1
            i = j
            continue
        if ln == "Check on examples":
            in_check = True
            seen = 0
            extra = 0
            out.append(ln)
            i += 1
            continue
        m = CHECK_RE.match(ln) if in_check else None
        if m:
            seen += 1
            if cap_ex and seen > cap_ex:
                extra += 1
                i += 1
                continue
            if compact_all:
                out.append(f"{m.group(1)} -> {m.group(2)}")
            else:
                out.append(ln)
            i += 1
            continue
        if in_check and ln.startswith("Apply to"):
            in_check = False
            if cap_ex and extra:
                out.append(f"(+{extra} more examples match)")
        out.append(ln)
        i += 1
    return "\n".join(out)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = [json.loads(l) for l in open('bit_real_all.jsonl', encoding='utf-8')]
    variants = {"T0_current": (0, 0, 0),
                "T1_dropSelected": (1, 0, 0),
                "T2_compactAll": (0, 1, 0),
                "T3_cap5": (0, 0, 5),
                "T4_all": (1, 1, 5)}
    stats = {k: {"toolong": 0, "lens": []} for k in variants}
    ww_share = []
    n_ok = 0
    rem_t4 = {"match": 0, "disagree": 0, "noprog": 0}
    for r in rows:
        p = Problem(id=r['id'], category='bit_manipulation',
                    examples=[Example(e['input_value'], e['output_value']) for e in r['examples']],
                    question=r['question'], answer=r['answer'])
        try:
            cot = Z.reasoning_bit_manipulation(p)
        except Exception:
            continue
        if cot is None or cot.rsplit('boxed{', 1)[-1].split('}')[0] != r['answer']:
            continue
        n_ok += 1
        # ww-section share of the full completion
        if "Whole-word check" in cot:
            head, ww = cot.split("Whole-word check", 1)
            ww_tok = len(tok.encode("Whole-word check" + ww, add_special_tokens=False))
            tot = len(tok.encode(cot, add_special_tokens=False))
            ww_share.append((ww_tok, tot))
        verdict = ('disagree' if 'Disagrees with the per-bit output' in cot
                   else 'noprog' if 'No agreeing program' in cot else 'match')
        for name, (ds, ca, ce) in variants.items():
            c2 = transform(cot, ds, ca, ce) if (ds or ca or ce) else cot
            completion = f"{c2.rstrip()}\n</think>\n\\boxed{{{r['answer']}}}<|im_end|>"
            n = len(tok.encode(completion, add_special_tokens=False)) + 4
            stats[name]["lens"].append(n)
            if n > CAP:
                stats[name]["toolong"] += 1
                if name == "T4_all":
                    rem_t4[verdict] += 1
    print(f"narrator-ok reals: {n_ok}")
    if ww_share:
        sh = sorted(w / t for w, t in ww_share)
        wt = sorted(w for w, _ in ww_share)
        m = len(sh)
        print(f"whole-word section: tok p50={wt[m//2]} p90={wt[int(m*.9)]} max={wt[-1]}; "
              f"share of CoT p50={sh[m//2]:.1%} p90={sh[int(m*.9)]:.1%}")
    for name in variants:
        d = sorted(stats[name]["lens"])
        m = len(d)
        print(f"{name:18s} toolong {stats[name]['toolong']:4d}  "
              f"p50={d[m//2]} p90={d[int(m*.9)]} max={d[-1]}")
    print("T4 remaining-toolong by verdict:", rem_t4)


if __name__ == '__main__':
    main()
