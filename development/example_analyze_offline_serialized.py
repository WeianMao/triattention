"""离线 DeepThinkLLM 结果分析（支持 msgpack 压缩格式）"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from development.serialization_utils import SerializationError, load_msgpack
from vllm.logprobs import Logprob

ALLOWED_SUFFIXES = (".msgpack.gz", ".msgpack.zst", ".msgpack", ".pkl", ".pickle")


def _has_allowed_suffix(path: Path) -> bool:
    name = path.name
    return any(name.endswith(ext) for ext in ALLOWED_SUFFIXES)


def find_result_files(output_dir: str, max_qid: int = None, rids: List[str] = None) -> List[Path]:
    """Find serialized offline result files in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    candidates = [
        path for path in output_path.glob("deepthink_offline_qid*_rid*_*")
        if path.is_file() and _has_allowed_suffix(path)
    ]

    filtered: List[Tuple[int, Path]] = []
    for path in candidates:
        qid, rid = extract_qid_rid(path.name)
        if qid is None or rid is None:
            continue
        if max_qid is not None and qid > max_qid:
            continue
        if rids and rid not in rids:
            continue
        filtered.append((qid, path))

    return [p for _, p in sorted(filtered, key=lambda item: item[0])]

def extract_qid_rid(filename: str) -> Tuple[int, str]:
    """Extract qid and rid from offline result filename."""
    import re

    pattern = r"deepthink_offline_qid(\d+)_rid(.+?)_\d{8}"
    match = re.search(pattern, filename)

    if match:
        return int(match.group(1)), match.group(2)
    return None, None

def load_result(filepath: Path) -> Dict:
    """Load a single result file (pickle or msgpack)."""
    suffixes = ''.join(filepath.suffixes)
    try:
        if suffixes.endswith(('.pkl', '.pickle')):
            with open(filepath, 'rb') as handle:
                result = pickle.load(handle)
        else:
            result = load_msgpack(str(filepath))
    except (SerializationError, Exception) as exc:  # noqa: BLE001
        print(f"Error loading {filepath}: {exc}")
        return None

    if isinstance(result, dict) and result.get('store_logprobs'):
        traces = result.get('all_traces') or []
        if traces:
            first_trace = traces[0]
            candidates = first_trace.get('logprobs') or []
            if candidates:
                sample = next(iter(candidates[0].values())) if isinstance(candidates[0], dict) else None
                if sample is not None and not isinstance(sample, Logprob):
                    print(f"Warning: logprob objects for {filepath.name} are not restored as vLLM Logprob")
    return result

def check_missing_files(output_dir: str, max_qid: int, rids: List[str]) -> Dict:
    """Check for missing qid/rid combinations in offline outputs."""
    output_path = Path(output_dir)

    if not output_path.exists():
        return {
            'total_expected': (max_qid + 1) * len(rids),
            'total_found': 0,
            'missing_count': (max_qid + 1) * len(rids),
            'missing_pairs': [(qid, rid) for qid in range(max_qid + 1) for rid in rids],
            'existing_pairs': [],
            'file_map': {}
        }

    existing_files = find_result_files(output_dir)
    existing_pairs = set()
    file_map: Dict[Tuple[int, str], str] = {}

    for filepath in existing_files:
        qid, rid = extract_qid_rid(filepath.name)
        if qid is None or rid is None or rid not in rids or qid > max_qid:
            continue
        existing_pairs.add((qid, rid))
        file_map[(qid, rid)] = filepath.name

    missing_pairs = [
        (qid, rid)
        for qid in range(max_qid + 1)
        for rid in rids
        if (qid, rid) not in existing_pairs
    ]

    return {
        'total_expected': (max_qid + 1) * len(rids),
        'total_found': len(existing_pairs),
        'missing_count': len(missing_pairs),
        'missing_pairs': sorted(missing_pairs),
        'existing_pairs': sorted(existing_pairs),
        'file_map': file_map
    }

def analyze_token_usage(results: List[Dict]) -> Dict:
    """Analyze token usage across all results"""
    token_stats = {
        'total_tokens': [],
        'tokens_per_trace': [],
        'budget': []
    }
    
    for result in results:
        if result:
            tokens = result.get('token_stats', {}).get('total_tokens', 0)
            token_stats['total_tokens'].append(tokens)

            budget = result.get('config', {}).get('budget')
            if budget is not None:
                token_stats['budget'].append(budget)

            total_traces = result.get('total_traces_count')
            if not total_traces:
                total_traces = len(result.get('all_traces', []))
            if total_traces:
                token_stats['tokens_per_trace'].append(tokens / total_traces)

    # Calculate statistics
    stats = {}
    for key, values in token_stats.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'total': np.sum(values)
            }
    
    return stats

def analyze_timing_details(results: List[Dict]) -> Dict:
    """Analyze detailed timing information from all results"""
    timing_components = {
        'total_time': [],
        'generation_time': [],
        'processing_time': [],
        'tokenizer_init_time': [],
        'llm_init_time': [],
    }
    
    # Derived timing metrics
    derived_metrics = {
        'init_total_time': [],       # tokenizer_init + llm_init
        'inference_time': [],        # total - init times
        'throughput_tokens_per_sec': [],
    }
    
    for result in results:
        if not result:
            continue
            
        timing_stats = result.get('timing_stats', {})
        token_stats = result.get('token_stats', {})
        
        # Basic timing components
        for component in timing_components:
            timing_components[component].append(timing_stats.get(component, 0))

        # Calculate derived metrics
        tokenizer_init = timing_stats.get('tokenizer_init_time', 0)
        llm_init = timing_stats.get('llm_init_time', 0)
        total_time = timing_stats.get('total_time', 0)

        derived_metrics['init_total_time'].append(tokenizer_init + llm_init)
        derived_metrics['inference_time'].append(max(0, total_time - tokenizer_init - llm_init))
        
        # Throughput calculations
        total_tokens = token_stats.get('total_tokens', 0)
        
        total_gen_time = timing_stats.get('generation_time', 0)
        if total_gen_time > 0 and total_tokens > 0:
            derived_metrics['throughput_tokens_per_sec'].append(total_tokens / total_gen_time)
        else:
            derived_metrics['throughput_tokens_per_sec'].append(0)
    
    # Combine all timing data
    all_timing_data = {**timing_components, **derived_metrics}
    
    # Calculate statistics
    timing_stats = {}
    for key, values in all_timing_data.items():
        if values:
            if 'throughput' in key.lower():
                values = [v for v in values if v > 0]
            
            if values:  # Check again after filtering
                timing_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'total': np.sum(values),
                    'count': len(values)
                }
    
    return timing_stats

def analyze_voting_methods(results: List[Dict]) -> Dict:
    """Analyze accuracy of different voting methods"""
    method_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'answers': [], 'num_votes': []})
    
    for result in results:
        if result and result.get('evaluation'):
            for method, eval_data in result['evaluation'].items():
                if eval_data.get('answer') is not None:
                    method_stats[method]['total'] += 1
                    if eval_data.get('is_correct'):
                        method_stats[method]['correct'] += 1
                    method_stats[method]['answers'].append({
                        'answer': eval_data['answer'],
                        'confidence': eval_data.get('confidence'),
                        'is_correct': eval_data.get('is_correct')
                    })
                    method_stats[method]['num_votes'].append(eval_data['num_votes'])
    
    # Calculate accuracy for each method
    method_accuracy = {}
    for method, stats in method_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            method_accuracy[method] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'avg_confidence': np.mean([a['confidence'] for a in stats['answers'] 
                                          if a['confidence'] is not None]) 
                                 if any(a['confidence'] for a in stats['answers']) else None,
                'num_votes' : stats['num_votes'] # np.mean(stats['num_votes']),
            }
    
    return method_accuracy

def analyze_confidence_methods(results: List[Dict]) -> Dict:
    """Analyze confidence-based evaluation methods"""
    conf_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for result in results:
        if result and result.get('confidence_evaluation'):
            for method, stats in result['confidence_evaluation'].items():
                conf_stats[method]['correct'] += stats.get('correct', 0)
                conf_stats[method]['total'] += stats.get('total', 0)
    
    # Calculate accuracy
    conf_accuracy = {}
    for method, stats in conf_stats.items():
        if stats['total'] > 0:
            conf_accuracy[method] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    return conf_accuracy

def print_timing_breakdown(timing_stats: Dict):
    """Print detailed timing breakdown"""
    print(f"\n⏱️ Detailed Timing Analysis")
    print("="*80)
    
    # Group timing metrics by category
    categories = {
        'Initialization Times': ['tokenizer_init_time', 'llm_init_time', 'init_total_time'],
        'Generation & Processing': ['generation_time', 'processing_time', 'total_time', 'inference_time'],
        'Throughput (tokens/sec)': ['throughput_tokens_per_sec'],
    }
    
    for category, metrics in categories.items():
        print(f"\n📊 {category}")
        print("-" * 60)
        
        # Header
        print(f"{'Metric':<25} {'Mean ± Std':<18} {'Median':<10} {'Range':<15}")
        print("-" * 70)
        
        for metric in metrics:
            if metric in timing_stats:
                stats = timing_stats[metric]
                
                # Format based on metric type
                if 'throughput' in metric.lower():
                    mean_str = f"{stats['mean']:.1f} ± {stats['std']:.1f}"
                    median_str = f"{stats['median']:.1f}"
                    range_str = f"[{stats['min']:.1f}, {stats['max']:.1f}]"
                else:
                    mean_str = f"{stats['mean']:.2f}s ± {stats['std']:.2f}s"
                    median_str = f"{stats['median']:.2f}s"
                    range_str = f"[{stats['min']:.2f}s, {stats['max']:.2f}s]"
                
                # Clean up metric name for display
                display_name = metric.replace('_', ' ').title()
                if len(display_name) > 24:
                    display_name = display_name[:21] + "..."
                
                print(f"{display_name:<25} {mean_str:<18} {median_str:<10} {range_str:<15}")
    
def print_statistics(token_stats: Dict, method_accuracy: Dict, conf_accuracy: Dict, 
                    missing_info: Dict, results: List[Dict], timing_stats: Dict = None):
    """Print comprehensive statistics"""
    print("\n" + "="*80)
    print("DEEPTHINK RESULTS ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 Overall Statistics")
    print("-"*40)
    print(f"Total result files analyzed: {len(results)}")
    valid_results = sum(1 for r in results if r is not None)
    print(f"Valid results: {valid_results}")
    
    # Token usage statistics
    if token_stats:
        print("\n💰 Token Usage Statistics")
        print("-"*40)
        for token_type, stats in token_stats.items():
            if not stats:
                continue
            header = token_type.replace('_', ' ').title()
            print(f"\n{header}:")
            if token_type == 'tokens_per_trace':
                print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"  Median: {stats['median']:.2f}")
                print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            else:
                print(f"  Mean: {stats['mean']:,.0f} ± {stats['std']:,.0f}")
                print(f"  Median: {stats['median']:,.0f}")
                print(f"  Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]")
                if 'total' in stats:
                    print(f"  Total: {stats['total']:,.0f}")
    
    # Detailed timing analysis
    if timing_stats:
        print_timing_breakdown(timing_stats)
    
    # Voting methods accuracy
    if method_accuracy:
        print(f"\n🗳️ Voting Methods Accuracy")
        print("-"*40)
        print(f"{'Method':<30} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Conf':<10}")
        print("-"*80)

        
        # Sort by accuracy
        sorted_methods = sorted(method_accuracy.items(), 
                              key=lambda x: x[1]['accuracy'], reverse=True)
        
        for method, stats in sorted_methods:
            acc_str = f"{stats['accuracy']:.1%}"
            ratio_str = f"{stats['correct']}/{stats['total']}"
            conf_str = f"{stats['avg_confidence']:.3f}" if stats['avg_confidence'] else "N/A"
            print(f"{method:<30} {acc_str:<12} {ratio_str:<15} {conf_str:<10}")
    
    # Confidence-based methods accuracy
    if conf_accuracy:
        print(f"\n🎯 Confidence-Based Methods")
        print("-"*40)
        print(f"{'Method':<25} {'Accuracy':<12} {'Correct/Total':<15}")
        print("-"*52)
        
        for method, stats in sorted(conf_accuracy.items(), 
                                   key=lambda x: x[1]['accuracy'], reverse=True):
            acc_str = f"{stats['accuracy']:.1%}"
            ratio_str = f"{stats['correct']}/{stats['total']}"
            print(f"{method:<25} {acc_str:<12} {ratio_str:<15}")
    
    # Missing files information
    print(f"\n📁 File Coverage Analysis")
    print("-"*40)
    print(f"Expected files: {missing_info['total_expected']}")
    print(f"Found files: {missing_info['total_found']}")
    print(f"Missing files: {missing_info['missing_count']}")
    coverage = (missing_info['total_found'] / missing_info['total_expected'] * 100 
                if missing_info['total_expected'] > 0 else 0)
    print(f"Coverage: {coverage:.1f}%")
    
    if missing_info['missing_pairs']:
        print(f"\n⚠️ Missing (qid, rid) pairs:")
        # Group by rid for better visualization
        by_rid = defaultdict(list)
        for qid, rid in missing_info['missing_pairs']:
            by_rid[rid].append(qid)
        
        for rid, qids in sorted(by_rid.items()):
            if len(qids) <= 10:
                print(f"  rid={rid}: qids={qids}")
            else:
                print(f"  rid={rid}: {len(qids)} missing qids (showing first 10): {qids[:10]}...")

def main():
    parser = argparse.ArgumentParser(description='Analyze DeepThinkLLM results with enhanced timing analysis')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing output pickle files')
    parser.add_argument('--max_qid', type=int, required=True,
                       help='Maximum question ID (0-based)')
    parser.add_argument('--rids', type=str, nargs='+', required=True,
                       help='List of run IDs to check')
    parser.add_argument('--force', action='store_true',
                       help='Force analysis even if files are missing')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check for missing files, do not analyze')
    parser.add_argument('--detailed_timing', action='store_true', default=True,
                       help='Enable detailed timing analysis (default: True)')
    
    args = parser.parse_args()
    
    # ========================================
    # PHASE 1: CHECK FOR MISSING FILES FIRST
    # ========================================
    print("\n" + "="*80)
    print("PHASE 1: FILE COMPLETENESS CHECK")
    print("="*80)
    print(f"Directory: {args.output_dir}")
    print(f"Expected QIDs: 0 to {args.max_qid} (total: {args.max_qid + 1})")
    print(f"Expected RIDs: {args.rids} (total: {len(args.rids)})")
    print(f"Total expected files: {(args.max_qid + 1) * len(args.rids)}")
    
    # Check for missing files
    missing_info = check_missing_files(args.output_dir, args.max_qid, args.rids)
    
    # Display results
    print("\n📊 Results:")
    print(f"  ✓ Found: {missing_info['total_found']} files")
    print(f"  ✗ Missing: {missing_info['missing_count']} files")
    coverage = (missing_info['total_found'] / missing_info['total_expected'] * 100 
                if missing_info['total_expected'] > 0 else 0)
    print(f"  📈 Coverage: {coverage:.1f}%")
    
    # Handle missing files
    if missing_info['missing_count'] > 0:
        print("\n" + "="*80)
        print("⚠️  MISSING FILES DETECTED")
        print("="*80)
        
        # Group by rid for cleaner display
        by_rid = defaultdict(list)
        for qid, rid in missing_info['missing_pairs']:
            by_rid[rid].append(qid)
        
        # Display missing files by rid
        for rid, qids in sorted(by_rid.items()):
            print(f"\nRID '{rid}':")
            print(f"  Missing {len(qids)} files")
            if len(qids) <= 15:
                print(f"  QIDs: {sorted(qids)}")
            else:
                print(f"  QIDs (first 15): {sorted(qids)[:15]}...")
                print(f"  ... and {len(qids)-15} more")
                
        # Exit or continue based on flags
        if args.check_only:
            print("\n✅ Check complete (--check_only flag used)")
            sys.exit(0)
        
        if not args.force:
            print("\n" + "="*80)
            print("❌ ABORTING: Files are missing")
            print("="*80)
            print("\nOptions:")
            print("1. Run missing experiments first")
            print("2. Use --force to analyze incomplete data")
            print("3. Use --check_only to only check for missing files")
            sys.exit(1)
        else:
            print("\n⚠️  WARNING: Continuing with incomplete data (--force used)")
            print("-"*80)
    else:
        print("\n✅ All expected files are present!")
        
        if args.check_only:
            print("✅ Check complete (--check_only flag used)")
            sys.exit(0)
    
    # ========================================
    # PHASE 2: LOAD AND ANALYZE FILES
    # ========================================
    print("\n" + "="*80)
    print("PHASE 2: LOADING AND ANALYZING FILES")
    print("="*80)
    
    # Find all result files
    result_files = find_result_files(args.output_dir, max_qid=args.max_qid, rids=args.rids)
    print(f"Loading {len(result_files)} files...")
    
    # Load results with progress indication
    results = []
    load_errors = []
    
    for i, filepath in tqdm(enumerate(result_files)):
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(result_files)} files...")
        
        result = load_result(filepath)
        if result:
            results.append(result)
        else:
            load_errors.append(filepath.name)
    
    # Report loading results
    print(f"\n📊 Loading Summary:")
    print(f"  ✓ Successfully loaded: {len(results)} files")
    if load_errors:
        print(f"  ✗ Failed to load: {len(load_errors)} files")
        if len(load_errors) <= 5:
            for err_file in load_errors:
                print(f"    - {err_file}")
        else:
            for err_file in load_errors[:5]:
                print(f"    - {err_file}")
            print(f"    ... and {len(load_errors)-5} more")
    
    if not results:
        print("\n❌ No valid results to analyze!")
        sys.exit(1)
    
    # ========================================
    # PHASE 3: ANALYZE RESULTS
    # ========================================
    print("\n" + "="*80)
    print("PHASE 3: ANALYZING RESULTS")
    print("="*80)
    
    # Perform analyses
    print("Running analyses...")
    token_stats = analyze_token_usage(results)
    method_accuracy = analyze_voting_methods(results)
    conf_accuracy = analyze_confidence_methods(results)
    
    # Detailed timing analysis
    timing_stats = None
    if args.detailed_timing:
        print("Analyzing detailed timing information...")
        timing_stats = analyze_timing_details(results)
    
    # Print comprehensive statistics
    print_statistics(token_stats, method_accuracy, conf_accuracy, missing_info, results, timing_stats)
        
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
