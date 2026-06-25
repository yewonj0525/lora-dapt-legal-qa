"""
Data Preparation for CaseHOLD Legal QA
Converts CaseHOLD CSV files to JSONL format for training.

CSV Format:
- Column 0: context (citing text)
- Columns 1-5: 5 candidate holdings
- Column 11: label (correct answer index 0-4)

Output JSONL Format:
{
    "id": "train_0",
    "context": "...",
    "endings": ["holding1", "holding2", "holding3", "holding4", "holding5"],
    "label": 2
}

Usage:
    python scripts/prepare_data.py \
        --input_dir casehold/data/all \
        --output_dir data/casehold
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm


def convert_csv_to_jsonl(csv_path, output_path, split_name):
    """
    Convert CaseHOLD CSV file to JSONL format.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        split_name: Name of the split (train/val/test)
    """
    print(f"\nProcessing {split_name}...")
    
    # Read CSV (no header in CaseHOLD files)
    df = pd.read_csv(csv_path, header=None)
    print(f"  Loaded {len(df)} examples")
    
    # Convert to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Converting {split_name}"):
            example = {
                'id': f"{split_name}_{idx}",
                'context': str(row[0]),  # Column 0: context
                'endings': [
                    str(row[1]),  # Column 1-5: 5 candidate holdings
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(row[5])
                ],
                'label': int(row[11])  # Column 11: correct answer (0-4)
            }
            f.write(json.dumps(example) + '\n')
    
    print(f"  ✓ Saved to {output_path}")
    return len(df)


def create_tiny_subset(input_path, output_path, n_samples=100):
    """
    Create a tiny subset for testing/debugging.
    
    Args:
        input_path: Path to full JSONL file
        output_path: Path to output tiny JSONL file
        n_samples: Number of samples to include
    """
    with open(input_path, 'r') as f_in:
        lines = f_in.readlines()[:n_samples]
    
    with open(output_path, 'w') as f_out:
        f_out.writelines(lines)
    
    print(f"  ✓ Created tiny subset: {output_path} ({n_samples} samples)")


def main(args):
    """Main function for data preparation."""
    
    print("="*60)
    print("CaseHOLD Data Preparation")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each split
    splits = ['train', 'dev', 'test']
    total_examples = {}
    
    for split in splits:
        csv_path = os.path.join(args.input_dir, f"{split}.csv")
        
        # dev -> val for consistency
        output_name = 'val' if split == 'dev' else split
        output_path = os.path.join(args.output_dir, f"{output_name}.jsonl")
        
        if os.path.exists(csv_path):
            n = convert_csv_to_jsonl(csv_path, output_path, output_name)
            total_examples[output_name] = n
        else:
            print(f"  Warning: {csv_path} not found, skipping...")
    
    # Create tiny subsets for testing
    print("\nCreating tiny subsets for testing...")
    for split in ['train', 'val', 'test']:
        input_path = os.path.join(args.output_dir, f"{split}.jsonl")
        output_path = os.path.join(args.output_dir, f"{split}_tiny.jsonl")
        if os.path.exists(input_path):
            create_tiny_subset(input_path, output_path, n_samples=100)
    
    # Print summary
    print("\n" + "="*60)
    print("Data Preparation Complete!")
    print("="*60)
    print("\nDataset Statistics:")
    for split, n in total_examples.items():
        print(f"  {split}: {n:,} examples")
    print(f"\nOutput directory: {args.output_dir}")
    
    # Verify data format
    print("\nVerifying data format...")
    sample_path = os.path.join(args.output_dir, "train.jsonl")
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            sample = json.loads(f.readline())
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Context length: {len(sample['context'])} chars")
        print(f"  Number of endings: {len(sample['endings'])}")
        print(f"  Label: {sample['label']}")
        print("  ✓ Data format verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CaseHOLD data")
    parser.add_argument('--input_dir', default='casehold/data/all', type=str,
                        help='Input directory containing CSV files')
    parser.add_argument('--output_dir', default='data/casehold', type=str,
                        help='Output directory for JSONL files')
    
    args = parser.parse_args()
    main(args)
