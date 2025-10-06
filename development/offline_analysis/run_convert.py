from pathlib import Path
from convert_offline_pickles import convert_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('pkl', type=Path)
parser.add_argument('--output-dir', type=Path, default=Path('development/offline_parsed'))
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

convert_file(args.pkl, args.output_dir, overwrite=args.overwrite)
