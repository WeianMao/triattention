"""快速加载 DeepConf 离线运行输出的辅助脚本。"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from development.serialization_utils import SerializationError, load_msgpack


def load_result(path: Path) -> Any:
    suffix = ''.join(path.suffixes)
    if suffix in {'.pkl', '.pickle'}:
        with path.open('rb') as f:
            return pickle.load(f)
    if suffix in {'.msgpack.gz', '.msgpack.zst', '.msgpack'}:
        return load_msgpack(str(path))
    raise SerializationError(f'Unsupported file extension: {suffix}')


def main() -> None:
    parser = argparse.ArgumentParser(description='DeepConf 离线结果加载工具')
    parser.add_argument('path', type=Path, help='结果文件路径（.pkl 或 .msgpack.*）')
    parser.add_argument('--export-pickle', type=Path,
                        help='可选，将加载结果重新导出为新的 pickle 文件')
    args = parser.parse_args()

    data = load_result(args.path)
    print(f'成功加载: {args.path}')
    print(f'顶层键: {sorted(data.keys())}')

    traces = data.get('result', {}).get('all_traces') or data.get('all_traces')
    if traces is not None:
        print(f'轨迹数量: {len(traces)}')

    if args.export_pickle:
        with args.export_pickle.open('wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'已导出 pickle: {args.export_pickle}')


if __name__ == '__main__':
    main()
