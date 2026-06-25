"""
Zero-shot Evaluation for CaseHOLD Legal QA
Evaluates pretrained RoBERTa-base without any fine-tuning.
Establishes the lower-bound baseline performance.

Usage:
    python scripts/eval_zeroshot.py --output_dir results/zeroshot
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class CaseHOLDDataset(Dataset):
    """
    Dataset class for CaseHOLD multiple-choice legal QA.
    Each example has a context and 5 candidate holdings.
    """
    def __init__(self, data_path, tokenizer, max_length=256):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        endings = item['endings']
        label = item['label']
        
        first_sentences = [context] * 5
        second_sentences = endings
        
        tokenized = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': torch.tensor(label)
        }


def collate_fn(batch):
    """Collate function to stack batch items."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def evaluate(model, dataloader, device):
    """
    Evaluate model on a dataset.
    Returns accuracy, F1 score, and average loss.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(dataloader)
    
    return {'accuracy': accuracy, 'f1': f1, 'loss': avg_loss}


def main(args):
    """Main function for zero-shot evaluation."""
    
    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load tokenizer and model (no training, just inference)
    print(f"\nLoading pretrained model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    
    # Load test dataset
    print(f"\nLoading test data: {args.test_data}")
    test_dataset = CaseHOLDDataset(args.test_data, tokenizer, args.max_length)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"Test examples: {len(test_dataset)}")
    
    # Evaluate
    print("\nRunning zero-shot evaluation...")
    test_metrics = evaluate(model, test_loader, device)
    
    # Save results
    results = {
        'method': 'zero_shot',
        'model_name': args.model_name,
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'test_loss': float(test_metrics['loss']),
        'trainable_params': 0,
        'total_params': total_params,
        'gpu_hours_estimate': 0,
        'note': 'No training - pretrained model only'
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ZERO-SHOT RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"\n✓ Results saved to {args.output_dir}/results.json")
    print(f"{'='*60}\n")
    
    # Note: Expected to be near random (20% for 5-way classification)
    print("Note: Zero-shot performance is expected to be near random")
    print("      (20% for 5-way classification) since the model")
    print("      has not been trained on this task.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation for CaseHOLD")
    parser.add_argument('--model_name', default='roberta-base', type=str,
                        help='Pretrained model name')
    parser.add_argument('--test_data', default='data/casehold/test.jsonl', type=str,
                        help='Path to test data')
    parser.add_argument('--output_dir', default='results/zeroshot', type=str,
                        help='Output directory for results')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Evaluation batch size')
    parser.add_argument('--max_length', default=256, type=int,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    main(args)
