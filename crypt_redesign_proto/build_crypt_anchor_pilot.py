"""v4 anchor pilot dataset build (cluster-side, mirrors build_crypt_pilot2.py
protocol with the \\b escape bug fixed: pilot2's plain-string "\boxed" put a
literal backspace in PROMPT_SUFFIX and in every completion's final answer line).

Composition: real cryptarithm_deduce rendered by crypt_anchor (kept on
boxed==gold), ~50% of emitting reals -> val pool (same md5 parity as pilot2),
train cap 100 real + 200 stratified synth (100 std / 50 swap / 30 easy /
20 concat) from synth_crypt_anchor.jsonl.

Output: crypt-anchor-pilot/{tokens/<pid>/synthetic.json, logprobs/index.jsonl,
val.jsonl} - same layout train_huikang_style_clipped.py + eval_adapter.py use.
Self-checks: no control chars in any completion; decoded tail of the first
record must contain "</think>" followed by "\\boxed{".
"""
import json, csv, os, sys, hashlib
import multiprocessing as mp
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
OUT = B + '/nemotron-master/training/sft/crypt-anchor-pilot'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
SYNTH_PICK = {'std': 100, 'swap': 50, 'easy': 30, 'concat': 20}

_tok = _nar = None


def _init():
    global _tok, _nar
    os.environ.setdefault('NEMOTRON_TOKENIZER_JSON', MODEL + '/tokenizer.json')
    # loud failure: a missing tokenizer silently degrades the narrator's exact
    # governor to a chars/1.40 estimate, shifting coverage vs the audited 160/659
    assert os.path.isfile(os.environ['NEMOTRON_TOKENIZER_JSON']), \
        os.environ['NEMOTRON_TOKENIZER_JSON']
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.crypt_anchor as A
    _nar = A.reasoning_cryptarithm_anchor


def work(task):
    from reasoners.store_types import Problem, Example
    src, pid, prompt, exs, ans, q = task
    prob = Problem(id=pid, category='cryptarithm_deduce',
                   examples=[Example(e['input_value'], e['output_value']) for e in exs],
                   question=q, answer=ans)
    try:
        cot = _nar(prob)
    except Exception:
        return None
    if cot is None:
        return None
    i = cot.rfind('boxed{'); j = cot.rfind('}')
    if not (0 <= i < j) or cot[i + 6:j] != ans:
        return None
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    assert not any(ord(c) < 9 for c in completion), f"control char in completion {pid}"
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return ('toolong', pid)
    return (src, pid, {'tokens': pids_ + cids, 'mask': [0] * len(pids_) + [1] * len(cids),
                       'nloss': len(cids)})


def is_val(pid):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2 == 0


def bucket_of(meta):
    if meta['tail'] == 'concat':
        return 'concat'
    if meta['tok'] < 1200:
        return 'easy'
    return meta['tail']


if __name__ == '__main__':
    meta = {json.loads(l)['id']: json.loads(l)['category']
            for l in open(B + '/nemotron-master/problems.jsonl')}
    trcsv = {}
    with open(B + '/nemotron-master/train.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            trcsv[r['id']] = (r['prompt'], str(r['answer']))
    real_pids = sorted(p for p, c in meta.items()
                       if c == 'cryptarithm_deduce' and p in trcsv)

    picked, counts = [], {k: 0 for k in SYNTH_PICK}
    for l in open(B + '/synth_crypt_anchor.jsonl', encoding='utf-8'):
        s = json.loads(l)
        b = bucket_of(s['_meta'])
        if counts.get(b, 99999) < SYNTH_PICK[b]:
            counts[b] += 1
            picked.append(s)
    assert counts == SYNTH_PICK, f"synth pick short: {counts}"

    tasks = []
    for pid in real_pids:
        d = json.load(open(B + f'/nemotron-master/problems/{pid}.jsonl'))
        prompt, ans = trcsv[pid]
        tasks.append(('real', pid, prompt, d['examples'], ans, str(d['question'])))
    for s in picked:
        tasks.append(('synth', s['id'], s['prompt'], s['examples'], s['answer'], s['question']))

    os.makedirs(OUT + '/tokens', exist_ok=True)
    os.makedirs(OUT + '/logprobs', exist_ok=True)
    real_rows, synth_rows, val, toolong = [], [], [], 0
    with mp.Pool(16, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            if r is None:
                continue
            if r[0] == 'toolong':
                toolong += 1
                continue
            src, pid, tk = r
            if src == 'real' and is_val(pid):
                prompt, ans = trcsv[pid]
                val.append({'id': pid, 'category': 'cryptarithm_deduce', 'source': 'real',
                            'prompt': prompt, 'answer': ans})
                continue
            (real_rows if src == 'real' else synth_rows).append((pid, tk))

    # deterministic selection + ordering regardless of imap arrival order
    real_rows.sort(key=lambda t: t[0])
    synth_rows.sort(key=lambda t: t[0])
    val.sort(key=lambda v: v['id'])
    real_rows = real_rows[:100]

    # per-bucket synth accounting: picks were asserted BEFORE narration; verify
    # nothing eroded at the narration/gold/toolong stage
    bucket_by_id = {s['id']: bucket_of(s['_meta']) for s in picked}
    from collections import Counter
    sc = Counter(bucket_by_id[pid] for pid, _ in synth_rows)
    print('synth survived per bucket:', dict(sc))
    assert dict(sc) == SYNTH_PICK, f"synth bucket erosion: {dict(sc)} != {SYNTH_PICK}"

    recs = []
    for pid, tk in real_rows + synth_rows:
        os.makedirs(f'{OUT}/tokens/{pid}', exist_ok=True)
        json.dump({'tokens': tk['tokens'], 'mask': tk['mask']},
                  open(f'{OUT}/tokens/{pid}/synthetic.json', 'w'))
        recs.append({'epoch': 0, 'step': -1, 'problem_id': pid, 'segment': 'synthetic.jsonl',
                     'category': 'cryptarithm_deduce', 'num_loss_tokens': tk['nloss'],
                     'total_loss': 0.0, 'min_logprob': 0.0})
    import random
    random.Random(0).shuffle(recs)
    with open(OUT + '/logprobs/index.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(OUT + '/val.jsonl', 'w') as f:
        for v in val:
            f.write(json.dumps(v) + '\n')

    assert real_rows and val, f"empty build: real_train={len(real_rows)} val={len(val)}"
    # post-build format check on one record
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    d = json.load(open(f"{OUT}/tokens/{recs[0]['problem_id']}/synthetic.json"))
    tail = tok.decode(d['tokens'][-60:])
    assert '</think>' in tail and '\\boxed{' in tail.split('</think>')[-1], repr(tail)
    print('tail check OK:', repr(tail[-60:]))
    print(f'train={len(recs)} (real {len(real_rows)}) val={len(val)} toolong={toolong}')
    print('OUT', OUT)
    print('PILOT_BUILD_DONE')
