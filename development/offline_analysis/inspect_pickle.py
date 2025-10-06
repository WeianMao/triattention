import argparse
import pickle
from pathlib import Path


def inspect_pickle(path: Path) -> None:
    print(f"Loading {path} ...")
    with path.open('rb') as f:
        data = pickle.load(f)

    print(f"Object type: {type(data)}")
    if isinstance(data, dict):
        print(f"Top-level keys ({len(data)}): {sorted(data.keys())}")
        traces = data.get('all_traces') or []
        print(f"Trace count: {len(traces)}")
        print(f"store_logprobs: {data.get('store_logprobs')}")
        if traces:
            first = traces[0]
            print(f"First trace keys: {sorted(first.keys())}")
            print(f"Tokens in first trace: {first.get('num_tokens')}")
            print(f"Logprobs stored: {'logprobs' in first}")
            if 'logprobs' in first:
                print(f"Length of logprobs: {len(first['logprobs'])}")

        evaluation = data.get('evaluation')
        if evaluation is not None:
            print(f"Evaluation keys: {sorted(evaluation.keys())}")
            print(f"Evaluation summary: {evaluation}")
        else:
            print("Evaluation: None")

        voting_results = data.get('voting_results') or {}
        print(f"Voting methods: {sorted(voting_results.keys())}")
        majority = voting_results.get('majority')
        if majority:
            print(f"Majority result: {majority}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect DeepThink offline pickle structure")
    parser.add_argument("path", type=str, help="Path to pickle file")
    args = parser.parse_args()

    inspect_pickle(Path(args.path))
