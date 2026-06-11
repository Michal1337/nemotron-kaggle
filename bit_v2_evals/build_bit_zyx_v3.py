"""bit-zyx-v3-uniform dataset build (eden, SLURM short partition).

Same protocol as build_bit_zyx_v2.py with three fixes:
  1. narrator = the UNIFORM whole-word-check render (every CoT carries the
     same final Whole-word check section; no rare override branch);
  2. the \\b backspace escape bug fixed (v2/8020 trained '\\x08oxed' in every
     completion tail and prompt suffix);
  3. deterministic selection (sort before seeded shuffle) + control-char
     assert + post-build tail check.
Val split: SAME pids as 8020/v2 (md5 %% 5 == 0 over narrator-ok reals; the
uniform render preserves every boxed answer, so the 304-pid val set is
unchanged and results stay comparable).
Synth: re-narrates the SAME synth_bit_zyx.jsonl problems under the new render.
"""
import json, csv, os, sys, hashlib
import multiprocessing as mp
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
OUT = B + '/nemotron-master/training/sft/bit-zyx-v3-uniform'
SYNTH = B + '/synth_bit_zyx.jsonl'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
_tok = _nar = None


def _init():
    global _tok, _nar
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.bit_manipulation_zyx as Z
    _nar = Z.reasoning_bit_manipulation


def _tok_cot(prompt, cot, ans):
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    assert not any(ord(c) < 9 for c in completion), "control char in completion"
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return None
    return {'tokens': pids_ + cids, 'mask': [0] * len(pids_) + [1] * len(cids),
            'nloss': len(cids)}


def work(task):
    from reasoners.store_types import Problem, Example
    src, pid, prompt, exs, ans, q = task
    prob = Problem(id=pid, category='bit_manipulation',
                   examples=[Example(e['input_value'], e['output_value']) for e in exs],
                   question=q, answer=ans)
    try:
        cot = _nar(prob)
    except Exception:
        return None
    if cot is None:
        return None
    b = cot.rsplit('boxed{', 1)[-1].split('}')[0]
    if b != ans:
        return None
    kind = ('override' if 'Disagrees with the per-bit output' in cot
            else 'match' if 'Matches the per-bit output.' in cot
            else 'noprog')
    tk = _tok_cot(prompt, cot, ans)
    if tk is None:
        return ('toolong', src, pid)
    return (src, pid, prompt, ans, tk, kind)


def is_val(pid):  # SAME split as bit-zyx-solved-8020 / v2-realsynth
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 5 == 0


if __name__ == '__main__':
    meta = {json.loads(l)['id']: json.loads(l)['category']
            for l in open(B + '/nemotron-master/problems.jsonl')}
    trcsv = {}
    with open(B + '/nemotron-master/train.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            trcsv[r['id']] = (r['prompt'], str(r['answer']))
    real = sorted(p for p, c in meta.items() if c == 'bit_manipulation' and p in trcsv)
    tasks = []
    for pid in real:
        d = json.load(open(f'{PB}/{pid}.jsonl'))
        prompt, ans = trcsv[pid]
        tasks.append(('real', pid, prompt, d['examples'], ans, str(d['question'])))
    for s in (json.loads(l) for l in open(SYNTH)):
        tasks.append(('synth', s['id'], s['prompt'], s['examples'], s['answer'], s['question']))

    os.makedirs(OUT + '/tokens', exist_ok=True)
    os.makedirs(OUT + '/logprobs', exist_ok=True)
    train_rows, val, tl = [], [], 0
    from collections import Counter
    kinds = Counter()
    with mp.Pool(16, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            if r is None:
                continue
            if r[0] == 'toolong':
                tl += 1
                continue
            src, pid, prompt, ans, tk, kind = r
            if src == 'real' and is_val(pid):
                val.append({'id': pid, 'category': 'bit_manipulation', 'source': 'real',
                            'prompt': prompt, 'answer': ans})
                continue
            kinds[f'{src}:{kind}'] += 1
            train_rows.append((pid, src, tk))

    # deterministic regardless of imap arrival order
    train_rows.sort(key=lambda t: t[0])
    val.sort(key=lambda v: v['id'])
    recs = []
    for pid, src, tk in train_rows:
        os.makedirs(f'{OUT}/tokens/{pid}', exist_ok=True)
        json.dump({'tokens': tk['tokens'], 'mask': tk['mask']},
                  open(f'{OUT}/tokens/{pid}/synthetic.json', 'w'))
        recs.append({'epoch': 0, 'step': -1, 'problem_id': pid, 'segment': 'synthetic.jsonl',
                     'category': 'bit_manipulation', 'num_loss_tokens': tk['nloss'],
                     'total_loss': 0.0, 'min_logprob': 0.0})
    import random
    random.Random(0).shuffle(recs)
    with open(OUT + '/logprobs/index.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(OUT + '/val.jsonl', 'w') as f:
        for v in val:
            f.write(json.dumps(v) + '\n')

    nr = sum(1 for _, s, _ in train_rows if s == 'real')
    ns = len(train_rows) - nr
    # post-build tail check on one record
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d = json.load(open(f"{OUT}/tokens/{recs[0]['problem_id']}/synthetic.json"))
    tail = tok.decode(d['tokens'][-40:])
    assert '</think>' in tail and '\\boxed{' in tail.split('</think>')[-1], repr(tail)
    print('tail check OK:', repr(tail[-70:]))
    print('train kinds:', dict(kinds))
    print(f'train={len(recs)} (real {nr}, synth {ns}) val={len(val)} toolong={tl}')
    print('OUT', OUT)
    print('BIT_V3_DONE')
