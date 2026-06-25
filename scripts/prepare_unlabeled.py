"""
Prepare Unlabeled Corpus for DAPT
Extracts legal contexts from CaseHOLD training set to create
unlabeled corpora of different sizes for domain-adaptive pre-training.

Note: We only use training set contexts to avoid data leakage
with validation and test sets.

Output: Text files with documents separated by double newlines.

Usage:
    python scripts/prepare_unlabeled.py \
        --input_csv casehold/data/all/train.csv \
        --output_dir data/unlabeled \
        --sizes 500 1000 3000 10000
"""

import os
import argparse
import random
import pandas as pd


def extract_unique_contexts(csv_path):
    """
    Extract unique legal contexts from CaseHOLD CSV.
    
    Args:
        csv_path: Path to CaseHOLD train.csv
        
    Returns:
        List of unique context strings
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV (column 0 is the context)
    df = pd.read_csv(csv_path, header=None)
    print(f"  Total examples: {len(df)}")
    
    # Extract contexts (column 0)
    contexts = df[0].tolist()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_contexts = []
    for ctx in contexts:
        ctx_str = str(ctx).strip()
        if ctx_str and ctx_str not in seen:
            seen.add(ctx_str)
            unique_contexts.append(ctx_str)
    
    print(f"  Unique contexts: {len(unique_contexts)}")
    
    return unique_contexts


def create_corpus(contexts, size, output_path, seed=42):
    """
    Create an unlabeled corpus of specified size.
    
    Args:
        contexts: List of all available contexts
        size: Number of documents to include
        output_path: Path to output file
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample contexts (or use all if size > available)
    if size >= len(contexts):
        selected = contexts
        print(f"  Using all {len(contexts)} contexts (requested {size})")
    else:
        selected = random.sample(contexts, size)
        print(f"  Sampled {size} contexts from {len(contexts)} available")
    
    # Write to file (documents separated by double newlines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(selected))
    
    # Report file size
    file_size = os.path.getsize(output_path)
    print(f"  ✓ Saved to {output_path} ({file_size/1024:.1f} KB)")


def main(args):
    """Main function for corpus preparation."""
    
    print("="*60)
    print("Unlabeled Corpus Preparation for DAPT")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract unique contexts from training data
    contexts = extract_unique_contexts(args.input_csv)
    
    # Parse sizes
    sizes = [int(s) for s in args.sizes]
    
    # Create corpora of different sizes
    print(f"\nCreating corpora of sizes: {sizes}")
    
    for size in sizes:
        output_path = os.path.join(args.output_dir, f"legal_corpus_{size}.txt")
        print(f"\nCreating corpus with {size} documents...")
        create_corpus(contexts, size, output_path, seed=args.seed)
    
    # Print summary
    print("\n" + "="*60)
    print("Corpus Preparation Complete!")
    print("="*60)
    print(f"\nCreated {len(sizes)} corpora in {args.output_dir}:")
    for size in sizes:
        output_path = os.path.join(args.output_dir, f"legal_corpus_{size}.txt")
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  - legal_corpus_{size}.txt ({file_size/1024:.1f} KB)")
    
    print("\nNote: These corpora use only training set contexts")
    print("      to avoid data leakage with val/test sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare unlabeled corpus for DAPT")
    parser.add_argument('--input_csv', default='casehold/data/all/train.csv', type=str,
                        help='Path to CaseHOLD training CSV')
    parser.add_argument('--output_dir', default='data/unlabeled', type=str,
                        help='Output directory for corpus files')
    parser.add_argument('--sizes', nargs='+', default=['500', '1000', '3000', '10000'],
                        help='Corpus sizes to create')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
