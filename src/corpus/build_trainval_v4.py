"""trainval-v4 assembly (eden, SLURM short).

TRAIN (~14k):
  frozen  cipher/unit_conversion/gravity/numeral -> C-crypt tokens copied VERBATIM
  eq      all narratable real (no holdout) + eq_synth_v4 (train slice)
  bit     all narratable real (zyx-v3 render) + synth_bit_zyx (train slice)
  crypt_deduce  all narratable real (anchor-v5) + synth_crypt_anchor_v5 (train slice)
  crypt_guess   all narratable real (guess-fam) + crypt_guess_synth (train slice)
VAL (source-separable):
  solved cats (cipher/unit/gravity/numeral) : 200 real spares NOT in C-crypt train
  hard cats (eq x2/bit/crypt x2)            : 200 fresh synth (held out of train,
                                              id-disjoint) + 100 real-insample probes
Hygiene: rfind boxed extraction ({}-bearing answers), control-char assert,
exact-tokenizer governor env (bit/anchor), deterministic sort-then-shuffle index.
"""
import json, csv, os, sys, shutil
import multiprocessing as mp
from collections import Counter
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
CCRYPT = B + '/nemotron-master/training/sft/C-crypt/C-crypt'
OUT = B + '/nemotron-master/training/sft/trainval-v4'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
FROZEN = ['cipher', 'unit_conversion', 'gravity', 'numeral']
HARD = ['equation_numeric_deduce', 'equation_numeric_guess', 'bit_manipulation',
        'cryptarithm_deduce', 'cryptarithm_guess']
SYNTH = {'eq': B + '/eq_synth_v4.jsonl', 'bit': B + '/synth_bit_zyx.jsonl',
         'anchor': B + '/synth_crypt_anchor_v5.jsonl', 'guess': B + '/crypt_guess_synth.jsonl'}
VAL_SYNTH_PER = 200
VAL_REAL_PER = 200
INSAMPLE_PER = 100

_tok = _EQ = _Z = _A = _CGF = None


def _init():
    global _tok, _EQ, _Z, _A, _CGF
    os.environ.setdefault('NEMOTRON_TOKENIZER_JSON', MODEL + '/tokenizer.json')
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.equation_numeric as EQ
    import reasoners.bit_manipulation_zyx as Z
    import reasoners.crypt_anchor as A
    import reasoners.crypt_guess_fam as CGF
    _EQ, _Z, _A, _CGF = EQ, Z, A, CGF


def _narrate(cat, prob):
    if cat.startswith('equation_numeric'):
        return _EQ.reasoning_equation_numeric(prob, None, None)
    if cat == 'bit_manipulation':
        return _Z.reasoning_bit_manipulation(prob)
    if cat == 'cryptarithm_deduce':
        return _A.reasoning_cryptarithm_anchor(prob)
    if cat == 'cryptarithm_guess':
        return _CGF.reasoning_cryptarithm_guess_fam(prob)
    return None


def work(task):
    from reasoners.store_types import Problem, Example
    cat, kind, pid, prompt, exs, ans, q = task
    prob = Problem(id=pid, category=cat,
                   examples=[Example(e['input_value'], e['output_value']) for e in exs],
                   question=q, answer=ans)
    try:
        cot = _narrate(cat, prob)
    except Exception:
        return None
    if cot is None:
        return None
    i = cot.rfind('boxed{'); j = cot.rfind('}')
    if not (0 <= i < j) or cot[i + 6:j] != ans:
        return None
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    assert not any(ord(c) < 9 for c in completion), f"control char {pid}"
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return ('toolong', cat, kind, pid)
    return (cat, kind, pid, {'tokens': pids_ + cids,
                             'mask': [0] * len(pids_) + [1] * len(cids), 'nloss': len(cids)})


def main():
    meta = {json.loads(l)['id']: json.loads(l)['category']
            for l in open(B + '/nemotron-master/problems.jsonl')}
    trcsv = {}
    with open(B + '/nemotron-master/train.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            trcsv[r['id']] = (r['prompt'], str(r['answer']))
    os.makedirs(OUT + '/tokens', exist_ok=True)
    os.makedirs(OUT + '/logprobs', exist_ok=True)
    recs, val = [], []
    kept = Counter()

    # ---- 1) FROZEN: copy C-crypt tokens + index rows verbatim ----
    cc_idx = [json.loads(l) for l in open(CCRYPT + '/logprobs/index.jsonl')]
    cc_idx = [r for r in cc_idx if r.get('epoch', 0) == 0]
    cc_train_pids = {r['problem_id'] for r in cc_idx}
    frozen_pids = {c: set() for c in FROZEN}
    for r in cc_idx:
        if r['category'] in FROZEN:
            pid = r['problem_id']
            src = f"{CCRYPT}/tokens/{pid}"
            if os.path.isdir(src):
                shutil.copytree(src, f"{OUT}/tokens/{pid}", dirs_exist_ok=True)
                recs.append(dict(r))
                frozen_pids[r['category']].add(pid)
                kept[f'{r["category"]}:frozen'] += 1

    # ---- 2) build narration tasks (real no-holdout + synth train slice) ----
    tasks = []
    real_by_cat = {c: [] for c in HARD}
    for pid, cat in meta.items():
        if cat in HARD and pid in trcsv:
            real_by_cat[cat].append(pid)
    for cat in HARD:
        for pid in sorted(real_by_cat[cat]):
            d = json.load(open(f'{PB}/{pid}.jsonl'))
            prompt, ans = trcsv[pid]
            tasks.append((cat, 'real', pid, prompt, d['examples'], ans, str(d['question'])))

    def load(path):
        return [json.loads(l) for l in open(path, encoding='utf-8')]
    # adaptive val split: at most VAL_SYNTH_PER, and never more than 25% of the
    # set (so a small synth pool like crypt_guess doesn't starve its train slice)
    def split(rows):
        rows = sorted(rows, key=lambda s: s['id'])
        n = min(VAL_SYNTH_PER, len(rows) // 4)
        return rows[:n], rows[n:]

    # eq synth: split per sub-category
    eq_rows = load(SYNTH['eq'])
    for sub in ('equation_numeric_deduce', 'equation_numeric_guess'):
        vh, tr = split([s for s in eq_rows if s['category'] == sub])
        for s in vh:
            val.append({'id': s['id'], 'category': sub, 'source': 'synth',
                        'prompt': s['prompt'], 'answer': s['answer']})
        for s in tr:
            tasks.append((sub, 'synth', s['id'], s['prompt'], s['examples'], s['answer'], s['question']))
    # bit / anchor / guess synth
    for key, cat in (('bit', 'bit_manipulation'), ('anchor', 'cryptarithm_deduce'),
                     ('guess', 'cryptarithm_guess')):
        vh, tr = split(load(SYNTH[key]))
        for s in vh:
            val.append({'id': s['id'], 'category': cat, 'source': 'synth',
                        'prompt': s['prompt'], 'answer': s['answer']})
        for s in tr:
            tasks.append((cat, 'synth', s['id'], s['prompt'], s['examples'], s['answer'], s['question']))

    # ---- 3) narrate + tokenize ----
    tl = 0
    train_real = {c: [] for c in HARD}
    with mp.Pool(16, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            if r is None:
                continue
            if r[0] == 'toolong':
                tl += 1
                continue
            cat, kind, pid, tk = r
            os.makedirs(f'{OUT}/tokens/{pid}', exist_ok=True)
            json.dump({'tokens': tk['tokens'], 'mask': tk['mask']},
                      open(f'{OUT}/tokens/{pid}/synthetic.json', 'w'))
            recs.append({'epoch': 0, 'step': -1, 'problem_id': pid, 'segment': 'synthetic.jsonl',
                         'category': cat, 'num_loss_tokens': tk['nloss'],
                         'total_loss': 0.0, 'min_logprob': 0.0})
            kept[f'{cat}:{kind}'] += 1
            if kind == 'real':
                train_real[cat].append(pid)

    # ---- 4) VAL real: solved-cat spares (not in C-crypt train) + insample probes ----
    real_pids_by_cat = {}
    for pid, cat in meta.items():
        real_pids_by_cat.setdefault(cat, []).append(pid)
    for cat in FROZEN + ['cipher']:
        spares = sorted(p for p in real_pids_by_cat.get(cat, [])
                        if p not in frozen_pids.get(cat, set()) and p not in cc_train_pids
                        and p in trcsv)
        for pid in spares[:VAL_REAL_PER]:
            prompt, ans = trcsv[pid]
            val.append({'id': pid, 'category': cat, 'source': 'real',
                        'prompt': prompt, 'answer': ans})
    for cat in HARD:
        for pid in sorted(train_real[cat])[:INSAMPLE_PER]:
            prompt, ans = trcsv[pid]
            val.append({'id': pid, 'category': cat, 'source': 'real-insample',
                        'prompt': prompt, 'answer': ans})

    # ---- write ----
    import random
    recs.sort(key=lambda r: r['problem_id'])
    random.Random(0).shuffle(recs)
    val.sort(key=lambda v: (v['category'], v['source'], v['id']))
    with open(OUT + '/logprobs/index.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(OUT + '/val.jsonl', 'w') as f:
        for v in val:
            f.write(json.dumps(v) + '\n')

    # ---- self-checks ----
    train_ids = {r['problem_id'] for r in recs}
    val_ids = {v['id'] for v in val}
    leak = train_ids & {v['id'] for v in val if v['source'] != 'real-insample'}
    assert not leak, f"VAL LEAK (non-insample val ids in train): {len(leak)}"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d = json.load(open(f"{OUT}/tokens/{recs[0]['problem_id']}/synthetic.json"))
    tail = tok.decode(d['tokens'][-40:])
    assert '</think>' in tail and '\\boxed{' in tail.split('</think>')[-1], repr(tail)
    print('tail check OK')
    print('train kept:', dict(sorted(kept.items())))
    print('train total:', len(recs), 'toolong:', tl)
    vc = Counter((v['category'], v['source']) for v in val)
    print('val:', dict(sorted(f"{c}/{s}" for (c, s) in vc)) if False else
          {f"{c}/{s}": n for (c, s), n in sorted(vc.items())})
    print('val total:', len(val))
    print('OUT', OUT)
    print('TRAINVAL_V4_DONE')


if __name__ == '__main__':
    main()
