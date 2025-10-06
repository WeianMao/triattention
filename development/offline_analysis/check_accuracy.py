import pickle
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=Path)
args = parser.parse_args()

with args.path.open('rb') as f:
    data = pickle.load(f)

ground_truth = str(data.get('ground_truth', '')).strip()
traces = data.get('all_traces', [])
total = len(traces)
correct = sum(1 for tr in traces if str(tr.get('extracted_answer', '')).strip() == ground_truth)
print(f'File: {args.path.name}')
print(f'Total traces: {total}')
print(f'Correct traces: {correct}')
print(f'Accuracy: {correct / total if total else 0:.4f}')
