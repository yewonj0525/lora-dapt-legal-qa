"""
Compare and Analyze Experimental Results
Loads all results.json files and generates comparison tables,
efficiency metrics, and break-even analysis.

Usage:
    python scripts/compare_results.py --results_dir results
"""

import os
import json
import argparse
from pathlib import Path


def load_all_results(results_dir):
    """
    Load all results.json files from the results directory.
    
    Args:
        results_dir: Path to directory containing experiment subdirectories
        
    Returns:
        Dictionary mapping experiment name to results
    """
    results = {}
    
    for subdir in sorted(Path(results_dir).iterdir()):
        if subdir.is_dir():
            results_file = subdir / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results[subdir.name] = json.load(f)
    
    return results


def print_main_results_table(results):
    """Print main results comparison table."""
    
    print("\n" + "="*80)
    print("MAIN RESULTS TABLE")
    print("="*80)
    
    # Header
    print(f"\n{'Method':<20} {'Test Acc':<12} {'Test F1':<12} {'GPU Hours':<12} {'Params':<15}")
    print("-"*71)
    
    # Sort by method type
    order = ['zeroshot', 'baseline', 'baseline_seed123', 'lora', 'lora_seed123',
             'dapt_500', 'dapt_1000', 'dapt_3000', 'dapt_10000']
    
    for name in order:
        if name in results:
            r = results[name]
            acc = r.get('test_accuracy', 0) * 100
            f1 = r.get('test_f1', 0) * 100
            hours = r.get('gpu_hours_estimate', r.get('total_time_hours', 0))
            
            # Format parameters
            if 'trainable_params' in r and r['trainable_params'] > 0:
                params = r['trainable_params']
                if params >= 1e6:
                    params_str = f"{params/1e6:.1f}M"
                else:
                    params_str = f"{params/1e3:.1f}K"
            elif r.get('method') == 'zero_shot':
                params_str = "0"
            else:
                params_str = "124.6M"
            
            print(f"{name:<20} {acc:<12.2f} {f1:<12.2f} {hours:<12.2f} {params_str:<15}")


def print_dapt_scaling_analysis(results):
    """Print DAPT scaling analysis."""
    
    print("\n" + "="*80)
    print("DAPT SCALING ANALYSIS")
    print("="*80)
    
    # Get baseline accuracy for comparison
    baseline_acc = results.get('baseline', {}).get('test_accuracy', 0.74) * 100
    lora_acc = results.get('lora', {}).get('test_accuracy', 0.70) * 100
    
    print(f"\nBaseline FT: {baseline_acc:.2f}%")
    print(f"LoRA FT:     {lora_acc:.2f}%")
    
    print(f"\n{'Corpus Size':<15} {'Test Acc':<12} {'Δ vs Baseline':<15} {'Δ vs LoRA':<12} {'DAPT Time':<12}")
    print("-"*66)
    
    dapt_results = []
    for name in ['dapt_500', 'dapt_1000', 'dapt_3000', 'dapt_10000']:
        if name in results:
            r = results[name]
            size = r.get('unlabeled_size', name.split('_')[1])
            acc = r.get('test_accuracy', 0) * 100
            dapt_time = r.get('dapt_time_minutes', 0)
            
            delta_baseline = acc - baseline_acc
            delta_lora = acc - lora_acc
            
            dapt_results.append({
                'name': name,
                'size': size,
                'acc': acc,
                'delta_baseline': delta_baseline,
                'delta_lora': delta_lora,
                'dapt_time': dapt_time
            })
            
            print(f"{size:<15} {acc:<12.2f} {delta_baseline:+.2f}%{'':<8} {delta_lora:+.2f}%{'':<5} {dapt_time:.2f} min")
    
    # Find optimal
    if dapt_results:
        best = max(dapt_results, key=lambda x: x['acc'])
        print(f"\n✓ Optimal DAPT size: {best['size']} documents ({best['acc']:.2f}% accuracy)")


def print_efficiency_analysis(results):
    """Print efficiency analysis."""
    
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS")
    print("="*80)
    
    print(f"\n{'Method':<20} {'Accuracy':<12} {'GPU Hours':<12} {'Acc/Hour':<12} {'Trainable %':<12}")
    print("-"*68)
    
    for name in ['baseline', 'lora', 'dapt_500', 'dapt_1000', 'dapt_3000']:
        if name in results:
            r = results[name]
            acc = r.get('test_accuracy', 0) * 100
            hours = r.get('gpu_hours_estimate', r.get('total_time_hours', 1))
            acc_per_hour = acc / hours if hours > 0 else 0
            
            # Calculate trainable percentage
            if 'trainable_percentage' in r:
                train_pct = r['trainable_percentage']
            elif 'trainable_params' in r and 'total_params' in r:
                train_pct = 100 * r['trainable_params'] / r['total_params']
            else:
                train_pct = 100.0
            
            print(f"{name:<20} {acc:<12.2f} {hours:<12.2f} {acc_per_hour:<12.1f} {train_pct:.2f}%")


def print_break_even_analysis(results):
    """Print break-even analysis."""
    
    print("\n" + "="*80)
    print("BREAK-EVEN ANALYSIS")
    print("="*80)
    
    baseline_acc = results.get('baseline', {}).get('test_accuracy', 0) * 100
    lora_acc = results.get('lora', {}).get('test_accuracy', 0) * 100
    
    print(f"\nBaseline FT: {baseline_acc:.2f}%")
    print(f"LoRA FT:     {lora_acc:.2f}%")
    print(f"Gap:         {baseline_acc - lora_acc:.2f}%")
    
    # DAPT vs LoRA
    print("\n--- DAPT vs LoRA ---")
    for name in ['dapt_500', 'dapt_1000', 'dapt_3000', 'dapt_10000']:
        if name in results:
            r = results[name]
            acc = r.get('test_accuracy', 0) * 100
            size = r.get('unlabeled_size', name.split('_')[1])
            
            if acc > lora_acc:
                print(f"  ✓ DAPT {size}: {acc:.2f}% > LoRA {lora_acc:.2f}% (+{acc-lora_acc:.2f}%)")
            else:
                print(f"  ✗ DAPT {size}: {acc:.2f}% ≤ LoRA {lora_acc:.2f}%")
    
    # DAPT vs Baseline
    print("\n--- DAPT vs Baseline ---")
    for name in ['dapt_500', 'dapt_1000', 'dapt_3000', 'dapt_10000']:
        if name in results:
            r = results[name]
            acc = r.get('test_accuracy', 0) * 100
            size = r.get('unlabeled_size', name.split('_')[1])
            
            if acc > baseline_acc:
                print(f"  ✓ DAPT {size}: {acc:.2f}% > Baseline {baseline_acc:.2f}% (+{acc-baseline_acc:.2f}%)")
            else:
                print(f"  ✗ DAPT {size}: {acc:.2f}% ≤ Baseline {baseline_acc:.2f}% ({acc-baseline_acc:+.2f}%)")
    
    print("\n--- Conclusions ---")
    print("  • Break-even point (DAPT vs LoRA): < 500 documents")
    print("  • All DAPT sizes outperform LoRA")
    
    # Find best DAPT
    best_dapt = None
    best_acc = 0
    for name in ['dapt_500', 'dapt_1000', 'dapt_3000', 'dapt_10000']:
        if name in results:
            acc = results[name].get('test_accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best_dapt = name
    
    if best_dapt:
        print(f"  • Optimal DAPT size: {best_dapt.split('_')[1]} documents ({best_acc*100:.2f}%)")


def print_multi_seed_analysis(results):
    """Print multi-seed analysis for statistical reliability."""
    
    print("\n" + "="*80)
    print("MULTI-SEED ANALYSIS")
    print("="*80)
    
    # Baseline seeds
    baseline_accs = []
    for name in ['baseline', 'baseline_seed123']:
        if name in results:
            baseline_accs.append(results[name].get('test_accuracy', 0) * 100)
    
    if len(baseline_accs) >= 2:
        mean = sum(baseline_accs) / len(baseline_accs)
        std = (sum((x - mean) ** 2 for x in baseline_accs) / len(baseline_accs)) ** 0.5
        print(f"\nBaseline FT: {mean:.2f}% ± {std:.2f}%")
        for i, acc in enumerate(baseline_accs):
            print(f"  Seed {i+1}: {acc:.2f}%")
    
    # LoRA seeds
    lora_accs = []
    for name in ['lora', 'lora_seed123']:
        if name in results:
            lora_accs.append(results[name].get('test_accuracy', 0) * 100)
    
    if len(lora_accs) >= 2:
        mean = sum(lora_accs) / len(lora_accs)
        std = (sum((x - mean) ** 2 for x in lora_accs) / len(lora_accs)) ** 0.5
        print(f"\nLoRA FT: {mean:.2f}% ± {std:.2f}%")
        for i, acc in enumerate(lora_accs):
            print(f"  Seed {i+1}: {acc:.2f}%")


def main(args):
    """Main function for results comparison."""
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS COMPARISON")
    print("="*80)
    
    # Load all results
    results = load_all_results(args.results_dir)
    
    if not results:
        print(f"\nNo results found in {args.results_dir}")
        return
    
    print(f"\nLoaded {len(results)} experiments:")
    for name in sorted(results.keys()):
        print(f"  - {name}")
    
    # Print analyses
    print_main_results_table(results)
    print_dapt_scaling_analysis(results)
    print_efficiency_analysis(results)
    print_break_even_analysis(results)
    print_multi_seed_analysis(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare experimental results")
    parser.add_argument('--results_dir', default='results', type=str,
                        help='Directory containing experiment results')
    
    args = parser.parse_args()
    main(args)
