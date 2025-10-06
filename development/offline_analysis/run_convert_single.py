from pathlib import Path
from convert_offline_pickles import convert_file

if __name__ == "__main__":
    pkl = Path('outputs/deepseek_r1_qwen3_8b/offline/deepthink_offline_qid0_rid0_20251002_182616.pkl')
    out = Path('development/offline_parsed')
    convert_file(pkl, out, overwrite=True)
    traces_dir = out / pkl.stem / 'traces'
    print('exists', traces_dir.exists())
    print('count', len(list(traces_dir.iterdir())) if traces_dir.exists() else 0)
