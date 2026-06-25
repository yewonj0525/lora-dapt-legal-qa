"""
Standard Fine-tuning for CaseHOLD Legal QA
Full fine-tuning of RoBERTa-base on the CaseHOLD multiple-choice task.

Usage:
    python scripts/train_baseline.py --output_dir results/baseline --epochs 3
"""

import os
import json
import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice,
    get_linear_schedule_with_warmup
)
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
        endings = item['endings']  # List of 5 candidate holdings
        label = item['label']  # Correct answer index (0-4)
        
        # Create 5 context-ending pairs for multiple choice
        first_sentences = [context] * 5
        second_sentences = endings
        
        # Tokenize all 5 pairs
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


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    """Main training function."""
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CaseHOLDDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = CaseHOLDDataset(args.val_data, tokenizer, args.max_length)
    test_dataset = CaseHOLDDataset(args.test_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")
    print(f"Test: {len(test_dataset)} examples")
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}")
    
    best_val_acc = 0
    training_stats = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        print("\nValidating...")
        val_metrics = evaluate(model, val_loader, device)
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Time: {elapsed_time/60:.2f} min")
        
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'time_minutes': elapsed_time / 60
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Saved best model (acc={best_val_acc:.4f})")
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    test_metrics = evaluate(model, test_loader, device)
    
    total_time = time.time() - start_time
    
    # Save results
    results = {
        'model_name': args.model_name,
        'method': 'baseline_ft',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'seed': args.seed,
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'test_loss': float(test_metrics['loss']),
        'total_time_minutes': total_time / 60,
        'total_time_hours': total_time / 3600,
        'training_stats': training_stats
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Total Time: {total_time/3600:.2f} hours")
    print(f"\n✓ Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard Fine-tuning for CaseHOLD")
    parser.add_argument('--model_name', default='roberta-base', type=str,
                        help='Pretrained model name')
    parser.add_argument('--train_data', default='data/casehold/train.jsonl', type=str,
                        help='Path to training data')
    parser.add_argument('--val_data', default='data/casehold/val.jsonl', type=str,
                        help='Path to validation data')
    parser.add_argument('--test_data', default='data/casehold/test.jsonl', type=str,
                        help='Path to test data')
    parser.add_argument('--output_dir', default='results/baseline', type=str,
                        help='Output directory for model and results')
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Training batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='Learning rate')
    parser.add_argument('--max_length', default=256, type=int,
                        help='Maximum sequence length')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    train(args)
