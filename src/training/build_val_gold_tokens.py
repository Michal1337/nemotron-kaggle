"""Held-out val gold-CoT tokenizer for the min-logprob GENERALIZATION probe.

The min-logprob objective needs per-token logprob of the CORRECT reasoning on
HELD-OUT problems. v4's held-out val rows store only {prompt, answer} (no CoT),
so this regenerates the gold CoT with the EXACT training narrator (deterministic,
honest) and tokenizes identically to training. Covers every narratable val pid:

  source=synth         -> held-out NEW problems (loaded from the synth source
                          jsonls, NOT the real problems dir). Narrated by the same
                          functions that built v4's train tokens => GENUINE
                          held-out generalization for eq/bit/crypt. (tier heldout-synth)
  source=real          -> out-of-sample real, frozen cats (gravity/numeral/unit).
                          Included ONLY if the repo narrator reproduces the
                          dataset's frozen TRAIN tokens (style match); a mismatched
                          style would give a misleading logprob. (tier heldout-real)
  source=real-insample -> in-train reals: a TRAIN-FIT baseline, NOT held-out.
                          (tier insample)

Writes  <ds>/val_gold_tokens/<pid>/synthetic.json   (tokens+mask, like training)
        <ds>/_val_gold_pids.txt                      (pids for score_adapter_logprobs)
        <ds>/_val_gold_manifest.jsonl                (pid,source,category,tier -> grouping)

Usage:  python build_val_gold_tokens.py --ds <dataset_dir>
"""
import argparse, json, os, sys
import multiprocessing as mp
from collections import Counter, defaultdict
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
# held-out synth problems live in their generator output, not the problems dir
SYNTH_SRC = {
    'equation_numeric_deduce': B + '/eq_synth_v4.jsonl',
    'equation_numeric_guess':  B + '/eq_synth_v4.jsonl',
    'bit_manipulation':        B + '/synth_bit_zyx.jsonl',
    'cryptarithm_deduce':      B + '/synth_crypt_anchor_v5.jsonl',
    'cryptarithm_guess':       B + '/crypt_guess_synth.jsonl',
}
FROZEN = ('cipher', 'gravity', 'numeral', 'unit_conversion')
TIER = {'synth': 'heldout-synth', 'real': 'heldout-real', 'real-insample': 'insample'}

_tok = None
_NAR = None


def _init():
    global _tok, _NAR
    os.environ.setdefault('NEMOTRON_TOKENIZER_JSON', MODEL + '/tokenizer.json')
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    import reasoners.equation_numeric as EQ
    import reasoners.bit_manipulation_zyx as Z
    import reasoners.crypt_anchor as A
    import reasoners.crypt_guess_fam as CGF
    import reasoners.cipher as CIP
    import reasoners.gravity as GR
    import reasoners.numeral as NU
    import reasoners.unit_conversion as UC
    _NAR = {
        'equation_numeric_deduce': lambda p: EQ.reasoning_equation_numeric(p, None, None),
        'equation_numeric_guess':  lambda p: EQ.reasoning_equation_numeric(p, None, None),
        'bit_manipulation':        Z.reasoning_bit_manipulation,
        'cryptarithm_deduce':      A.reasoning_cryptarithm_anchor,
        'cryptarithm_guess':       CGF.reasoning_cryptarithm_guess_fam,
        'cipher':                  CIP.reasoning_cipher,
        'gravity':                 GR.reasoning_gravity,
        'numeral':                 NU.reasoning_numeral,
        'unit_conversion':         UC.reasoning_unit_conversion,
    }


def _cot_tokens(pid, cat, prompt, ans, examples, question):
    """Narrate + gold-check + tokenize. Returns (tokens, mask, cids) or a str reason."""
    from reasoners.store_types import Problem, Example
    nar = _NAR.get(cat)
    if nar is None:
        return 'no_narrator'
    prob = Problem(id=pid, category=cat,
                   examples=[Example(e['input_value'], e['output_value']) for e in examples],
                   question=str(question), answer=ans)
    try:
        cot = nar(prob)
    except Exception:
        return 'narrate_exc'
    if cot is None:
        return 'narrate_none'
    i = cot.rfind('boxed{'); j = cot.rfind('}')
    if not (0 <= i < j) or cot[i + 6:j] != ans:
        return 'gold_miss'
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    if any(ord(c) < 9 for c in completion):
        return 'ctrl_char'
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': (prompt or '') + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pre = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return 'toolong'
    return (pre + cids, [0] * len(pre) + [1] * len(cids), cids)


def work_val(task):
    pid, cat, source, prompt, ans, examples, question = task
    r = _cot_tokens(pid, cat, prompt, ans, examples, question)
    if isinstance(r, str):
        return ('fail', source, cat, r, pid)
    tokens, mask, _ = r
    return ('ok', source, cat, pid, tokens, mask)


def work_check(task):
    """Narrate a frozen-cat TRAIN pid; does the completion match the stored frozen
    tokens? (CoT tokens are prompt-independent, so prompt is irrelevant here.)"""
    pid, cat, examples, question, stored_comp = task
    txt = _tok.decode(stored_comp)
    bi = txt.rfind('boxed{'); bj = txt.rfind('}')
    ans = txt[bi + 6:bj] if 0 <= bi < bj else None
    if ans is None:
        return (cat, False)
    r = _cot_tokens(pid, cat, '', ans, examples, question)
    if isinstance(r, str):
        return (cat, False)
    return (cat, r[2] == stored_comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    ap.add_argument('--check-sample', type=int, default=16,
                    help='frozen-cat train pids to test for narrator-style match')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    DS = args.ds
    out_tok = DS + '/val_gold_tokens'
    import shutil
    if os.path.isdir(out_tok):  # clean rebuild (drop any stale/invalidated pid dirs)
        shutil.rmtree(out_tok)
    os.makedirs(out_tok, exist_ok=True)
    val = [json.loads(l) for l in open(DS + '/val.jsonl')]

    # synth problem map (id -> {examples, question, answer})
    synth_map = {}
    for path in set(SYNTH_SRC.values()):
        if os.path.exists(path):
            for l in open(path):
                e = json.loads(l)
                synth_map[e['id']] = e

    # frozen-cat narrator self-check tasks (sample train pids from the index)
    idx = [json.loads(l) for l in open(DS + '/logprobs/index.jsonl')]
    frozen_train = defaultdict(list)
    for r in idx:
        if r['category'] in FROZEN:
            frozen_train[r['category']].append(r['problem_id'])
    check_tasks = []
    for cat, pids in frozen_train.items():
        for pid in sorted(pids)[:args.check_sample]:
            tf, pf = f'{DS}/tokens/{pid}/synthetic.json', f'{PB}/{pid}.jsonl'
            if not (os.path.exists(tf) and os.path.exists(pf)):
                continue
            td, d = json.load(open(tf)), json.load(open(pf))
            comp = [t for t, m in zip(td['tokens'], td['mask']) if m == 1]
            check_tasks.append((pid, cat, d['examples'], str(d['question']), comp))

    # build val narration tasks
    tasks, skipped = [], Counter()
    for v in val:
        pid, cat, src = v['id'], v['category'], v.get('source')
        ans, prompt = v.get('answer'), v.get('prompt')
        if src == 'synth':
            e = synth_map.get(pid)
            if e is None:
                skipped[(src, cat, 'no_synth_src')] += 1; continue
            examples, question = e['examples'], e.get('question', '')
            if ans in (None, ''):
                ans = str(e.get('answer'))
        else:
            pf = f'{PB}/{pid}.jsonl'
            if not os.path.exists(pf):
                skipped[(src, cat, 'no_problem')] += 1; continue
            d = json.load(open(pf))
            examples, question = d['examples'], str(d['question'])
        tasks.append((pid, cat, src, prompt, ans, examples, question))

    # run self-check first -> which frozen cats' held-out reals are style-valid
    passed_frozen = {}
    with mp.Pool(args.workers, initializer=_init) as pool:
        agg = defaultdict(lambda: [0, 0])
        for cat, ok in pool.imap_unordered(work_check, check_tasks, chunksize=2):
            agg[cat][0] += int(ok); agg[cat][1] += 1
        print('=== frozen-cat narrator self-check (repo narrator == frozen train tokens?) ===')
        for cat in FROZEN:
            hit, tot = agg.get(cat, [0, 0])
            ok = tot > 0 and hit == tot
            passed_frozen[cat] = ok
            print(f'  {cat:16s} match {hit}/{tot}  -> {"VALID held-out-real" if ok else "SKIP (style mismatch / no train pids)"}')

    # drop ALL val rows (real AND real-insample) for frozen cats that failed the
    # self-check: the repo narrator can't reproduce the trained format, so any
    # logprob there scores a CoT style the adapter never saw (cipher insample
    # included - it was trained on C-crypt-format tokens, not reasoning_cipher).
    kept_tasks = []
    for t in tasks:
        pid, cat, src = t[0], t[1], t[2]
        if cat in FROZEN and not passed_frozen.get(cat):
            skipped[(src, cat, 'style_mismatch')] += 1; continue
        kept_tasks.append(t)

    # narrate + tokenize val
    manifest, ok_pids = [], []
    fails = Counter()
    with mp.Pool(args.workers, initializer=_init) as pool:
        for res in pool.imap_unordered(work_val, kept_tasks, chunksize=4):
            if res[0] == 'fail':
                _, source, cat, reason, pid = res
                fails[(source, cat, reason)] += 1
                continue
            _, source, cat, pid, tokens, mask = res
            os.makedirs(f'{out_tok}/{pid}', exist_ok=True)
            json.dump({'tokens': tokens, 'mask': mask},
                      open(f'{out_tok}/{pid}/synthetic.json', 'w'))
            ok_pids.append(pid)
            manifest.append({'pid': pid, 'source': source, 'category': cat,
                             'tier': TIER.get(source, source)})

    ok_pids.sort()
    open(DS + '/_val_gold_pids.txt', 'w').write('\n'.join(ok_pids) + '\n')
    with open(DS + '/_val_gold_manifest.jsonl', 'w') as f:
        for m in sorted(manifest, key=lambda m: m['pid']):
            f.write(json.dumps(m) + '\n')

    # coverage report
    req = Counter((v.get('source'), v['category']) for v in val)
    got = Counter((m['source'], m['category']) for m in manifest)
    print('\n=== gold-token coverage by (tier / source / category) ===')
    for (src, cat) in sorted(req):
        print(f'  {TIER.get(src, src):14s} {src:14s} {cat:26s} {got.get((src, cat), 0):4d} / {req[(src, cat)]:4d}')
    by_tier = Counter(m['tier'] for m in manifest)
    print('\nnarrated by tier:', dict(by_tier))
    print('total narrated:', len(ok_pids), 'of', len(val), 'val rows')
    if fails:
        print('fail reasons:', {f'{s}/{c}/{r}': n for (s, c, r), n in fails.most_common()})
    if skipped:
        print('skipped (pre-narrate):', {f'{s}/{c}/{r}': n for (s, c, r), n in skipped.most_common()})
    print('VAL_GOLD_TOKENS_DONE')


if __name__ == '__main__':
    main()
