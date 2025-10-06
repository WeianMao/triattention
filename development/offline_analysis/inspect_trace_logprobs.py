import pickle
from pathlib import Path

fn = Path('outputs/deepseek_r1_qwen3_8b/offline/deepthink_offline_qid0_rid0_20251002_182616.pkl')
with fn.open('rb') as f:
    data = pickle.load(f)
trace = data['all_traces'][0]
logprobs = trace['logprobs']
token_ids = trace['token_ids']
print('len token_ids', len(token_ids))
print('len logprobs', len(logprobs))
first = logprobs[0]
print('type first', type(first))
keys = list(first.keys())
print('num keys first dict', len(keys))
print('first key repr', repr(keys[0]))
entry = first[keys[0]]
print('entry type', type(entry))
print('entry attributes', [a for a in dir(entry) if not a.startswith('_')])
print('entry logprob', entry.logprob)
print('entry decoded token', getattr(entry, 'decoded_token', None))
key_match = token_ids[0]
print('selected token id', key_match)
print('selected in dict', key_match in first)
if key_match in first:
    print('selected logprob', first[key_match].logprob)
print('done')
