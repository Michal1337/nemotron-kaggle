"""Build held-out val gold-CoT tokens for the eq adapter logprob test.
Narrate each of the 134 eq val problems (UNSEEN during training) with the same
narrator, tokenize identically to build_eq_8020.py, write tokens so
score_adapter_logprobs.py can score the adapter's per-token logprob on the
CORRECT reasoning for unseen problems = a generalization-relevant min-logprob.
"""
import json, os, sys
sys.path.insert(0, 'src')
B = '/mnt/evafs/groups/re-com/mgromadzki'
PB = B + '/nemotron-master/problems'
MODEL = B + '/llms/nemotron-3-nano-30b-a3b-bf16'
DS = B + '/nemotron-master/training/sft/eq-deduce-guess-8020'
OUT_TOK = DS + '/val_gold_tokens'
PIDS = DS + '/_val_gold_pids.txt'
PROMPT_SUFFIX = ("\nPlease put your final answer inside `\\boxed{}`. "
                 "For example: `\\boxed{your answer}`")

from transformers import AutoTokenizer
import reasoners.equation_numeric as EQ
from reasoners.store_types import Problem, Example

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
val = [json.loads(l) for l in open(DS + '/val.jsonl')]
os.makedirs(OUT_TOK, exist_ok=True)
pids, skipped = [], 0
for v in val:
    pid, prompt, ans, cat = v['id'], v['prompt'], v['answer'], v['category']
    d = json.load(open(f'{PB}/{pid}.jsonl'))
    prob = Problem(id=pid, category=cat,
                   examples=[Example(e['input_value'], e['output_value']) for e in d['examples']],
                   question=str(d['question']), answer=ans)
    cot = EQ.reasoning_equation_numeric(prob, None, None)
    if cot is None or cot.rsplit('boxed{', 1)[-1].split('}')[0] != ans:
        skipped += 1
        continue
    completion = f"{cot.rstrip()}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    rendered = tok.apply_chat_template(
        [{'role': 'user', 'content': prompt + PROMPT_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids_ = list(tok.encode(rendered, add_special_tokens=False))
    cids = list(tok.encode(completion, add_special_tokens=False))
    os.makedirs(f'{OUT_TOK}/{pid}', exist_ok=True)
    json.dump({'tokens': pids_ + cids, 'mask': [0] * len(pids_) + [1] * len(cids)},
              open(f'{OUT_TOK}/{pid}/synthetic.json', 'w'))
    pids.append(pid)
open(PIDS, 'w').write('\n'.join(pids) + '\n')
print(f'built val gold tokens: {len(pids)} (skipped {skipped} non-narratable)')
print('OUT_TOK', OUT_TOK, 'PIDS', PIDS)
