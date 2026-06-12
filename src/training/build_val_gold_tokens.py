"""Generic held-out val gold-CoT tokenizer for the min-logprob generalization
probe. Narrates the REAL val pids of a dataset (source=real / real-insample)
with the matching category narrator and tokenizes identically to training, so
score_adapter_logprobs can measure the adapter's per-token logprob on the
CORRECT reasoning for held-out problems. Usage:
  python build_val_gold_tokens.py --ds <dataset_dir>
writes <ds>/val_gold_tokens/<pid>/synthetic.json + <ds>/_val_gold_pids.txt
"""
import argparse, json, csv, os, sys
import multiprocessing as mp
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")
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
    pid, cat, prompt, ans = task
    try:
        d = json.load(open(f'{PB}/{pid}.jsonl'))
    except Exception:
        return None
    prob = Problem(id=pid, category=cat,
                   examples=[Example(e['input_value'], e['output_value']) for e in d['examples']],
                   question=str(d['question']), answer=ans)
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
    rendered = _tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(_tok.encode(rendered, add_special_tokens=False))
    cids = list(_tok.encode(completion, add_special_tokens=False))
    if len(cids) > 7680:
        return None
    return (pid, {'tokens': pids_ + cids, 'mask': [0] * len(pids_) + [1] * len(cids)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True)
    args = ap.parse_args()
    out_tok = args.ds + '/val_gold_tokens'
    os.makedirs(out_tok, exist_ok=True)
    val = [json.loads(l) for l in open(args.ds + '/val.jsonl')]
    # REAL val pids only (real / real-insample) - the held-out generalization set
    tasks = [(v['id'], v['category'], v['prompt'], v['answer'])
             for v in val if v.get('source') in ('real', 'real-insample')]
    pids = []
    with mp.Pool(16, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=4):
            if r is None:
                continue
            pid, tk = r
            os.makedirs(f'{out_tok}/{pid}', exist_ok=True)
            json.dump(tk, open(f'{out_tok}/{pid}/synthetic.json', 'w'))
            pids.append(pid)
    open(args.ds + '/_val_gold_pids.txt', 'w').write('\n'.join(pids) + '\n')
    print(f'val gold tokens: {len(pids)} narrated of {len(tasks)} real val pids')
    print('VAL_GOLD_TOKENS_DONE')
