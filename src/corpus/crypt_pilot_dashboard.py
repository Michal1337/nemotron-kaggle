"""Step-level dashboard for the crypt_vfirst pilot (audit build step 7).

Final accuracy alone cannot distinguish recall from procedure-learning (the
compact format scored 72% in-sample on pure recall). This dashboard parses the
MODEL'S generations and re-executes/cross-checks each procedural step against
ground truth computed by the narrator itself. Gates (from the merged plan):

  P0 rename fidelity   >= 99%   the generation's glyph->letter table is a
                                correct first-occurrence renaming
  arith validity       >= 98%   every arithmetic claim line in the generation
                                re-executes true (strict, gates G4 logic)
  self-exec            >= 98%   the boxed answer follows from the generation's
                                OWN final compute/encode lines
  op/mode id           >= 90%   the generation's chosen reading (combo line
                                after 'Resolve'/'DETERMINED') matches the
                                reference narrator's chosen reading
  per-tail accuracy             concat / det / agree-open, vs overall

Usage:
  python src/corpus/crypt_pilot_dashboard.py <eval-val.json>
  (eval_adapter output: {"results": [{id, gold, predicted, correct,
   generation, ...}]}; problems are read from the standard problems dir or
   crypt_real_all.jsonl when run locally.)
"""
import json
import os
import re
import sys

sys.path.insert(0, "src")
from corpus.crypt_vfirst_gates import check_arith_lines, boxed_of  # noqa: E402
from reasoners.crypt_vfirst import reasoning_cryptarithm_vfirst  # noqa: E402
from reasoners.store_types import Problem, Example  # noqa: E402

B = "/mnt/evafs/groups/re-com/mgromadzki"
# v2 interleaved P0: per-example "      <pairs>  ->  <letters>" lines
P0_LINE_RE = re.compile(r"->  ([A-Jpqr =]+)$", re.M)
ATTEMPT_RE = re.compile(r"^Attempt ([a-z_0-9.]+):$", re.M)
YIELDS_RE = re.compile(r"-> this reading yields")


def load_problem(pid):
    cand = f"{B}/nemotron-master/problems/{pid}.jsonl"
    if os.path.exists(cand):
        d = json.load(open(cand))
        return d["examples"], str(d["question"])
    for l in open("crypt_real_all.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["id"] == pid:
            return r["examples"], r["question"]
    raise KeyError(pid)


def true_rename(examples, question):
    """Reference first-occurrence renaming (mirrors crypt_vfirst.normalize)."""
    opchars, digmap = [], {}
    exs = [(e["input_value"], e["output_value"]) for e in examples]
    for inp, _ in exs:
        if inp[2] not in opchars:
            opchars.append(inp[2])
    if question[2] not in opchars:
        opchars.append(question[2])
    opmap = {c: "pqr"[i] for i, c in enumerate(opchars) if i < 3}

    def see(c):
        if c in opmap or c in digmap or len(digmap) >= 10:
            return
        digmap[c] = "ABCDEFGHIJ"[len(digmap)]

    for inp, out in exs:
        for c in (inp[0], inp[1], inp[3], inp[4]):
            see(c)
        body = out[1:] if (len(out) > 1 and out[0] == inp[2]) else out
        for c in body:
            see(c)
    for c in (question[0], question[1], question[3], question[4]):
        see(c)
    return {**opmap, **digmap}


def gen_rename_ok(gen, examples, question):
    """v2: the generation's renamed letter equations must equal the true
    renaming of every example + the query."""
    full = true_rename(examples, question)

    def ren(s):
        return "".join(full.get(c, c) for c in s)

    want = [f"{ren(e['input_value'])} = {ren(e['output_value'])}" for e in examples]
    want.append(ren(question))
    got = [x.strip() for x in P0_LINE_RE.findall(gen)]
    return got[:len(want)] == want


def chosen_reading(text):
    """The Attempt whose block contains '-> this reading yields'."""
    if not text:
        return None
    names = [(m.start(), m.group(1)) for m in ATTEMPT_RE.finditer(text)]
    if not names:
        return None
    y = YIELDS_RE.search(text)
    if not y:
        return None
    before = [nm for pos, nm in names if pos < y.start()]
    return before[-1] if before else None


def self_exec_ok(gen):
    """boxed must equal the encode result stated in the generation's own
    'Back to the original glyphs' / 'The answer is' tail (string-level
    consistency between the last encode mapping and the boxed value)."""
    b = boxed_of(gen)
    if b is None:
        return False
    tail = gen[gen.rfind("The answer is"):]
    return f"\\boxed{{{b}}}" in tail or f"boxed{{{b}}}" in tail


def tail_of(cot):
    if cot is None:
        return "abstain"
    if "no digit code needed" in cot:
        return "concat"
    if "UNDERDETERMINED:" in cot:
        return "under"
    if "DETERMINED:" in cot:
        return "det"
    if "resolved readings agree" in cot:
        return "agree"
    return "other"


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    rs = d["results"] if isinstance(d, dict) else d
    n = len(rs)
    m = {"p0": 0, "arith": 0, "selfexec": 0, "opmode": 0, "opmode_n": 0}
    tails = {}
    for r in rs:
        pid = r["id"]
        gen = r.get("generation") or ""
        examples, question = load_problem(pid)
        prob = Problem(id=pid, category="cryptarithm_deduce",
                       examples=[Example(e["input_value"], e["output_value"]) for e in examples],
                       question=question, answer=r["gold"])
        ref = reasoning_cryptarithm_vfirst(prob)
        t = tail_of(ref)
        tails.setdefault(t, [0, 0])
        tails[t][0] += 1
        tails[t][1] += bool(r.get("correct"))
        m["p0"] += gen_rename_ok(gen, examples, question)
        m["arith"] += not check_arith_lines(gen)
        m["selfexec"] += self_exec_ok(gen)
        ref_choice = chosen_reading(ref)
        if ref_choice is not None:
            m["opmode_n"] += 1
            m["opmode"] += (chosen_reading(gen) == ref_choice)
    print(f"n={n}")
    print(f"P0 rename fidelity : {m['p0']/n:.1%}   (gate >=99%)")
    print(f"arith validity     : {m['arith']/n:.1%}   (gate >=98%)")
    print(f"self-exec          : {m['selfexec']/n:.1%}   (gate >=98%)")
    if m["opmode_n"]:
        print(f"op/mode id         : {m['opmode']/m['opmode_n']:.1%} of {m['opmode_n']}   (gate >=90%)")
    print("per-tail accuracy  :")
    for t, (tn, tok) in sorted(tails.items()):
        print(f"  {t:8s} {tok}/{tn} = {tok/tn:.1%}")
    gates = [m['p0']/n >= 0.99, m['arith']/n >= 0.98, m['selfexec']/n >= 0.98,
             (m['opmode']/m['opmode_n'] >= 0.90) if m['opmode_n'] else True]
    print("PILOT GO" if all(gates) else "PILOT NO-GO (fix renderer/data, re-pilot; do not add epochs)")


if __name__ == "__main__":
    main()
