"""
DAPT (Domain-Adaptive Pre-Training) for CaseHOLD Legal QA
Two-phase training:
  Phase 1: Masked Language Modeling (MLM) on unlabeled legal corpus
  Phase 2: Fine-tuning on CaseHOLD multiple-choice task

Usage:
    python scripts/train_dapt.py \
        --unlabeled_corpus data/unlabeled/legal_corpus_1000.txt \
        --output_dir results/dapt_1000 \
        --dapt_epochs 1 --ft_epochs 3
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
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class UnlabeledTextDataset(Dataset):
    """
    Dataset for unlabeled legal text used in DAPT MLM phase.
    Documents are separated by double newlines.
    """
    def __init__(self, corpus_path, tokenizer, max_length=256):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')
        
        self.texts = [t.strip() for t in texts if t.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loaded {len(self.texts)} unlabeled documents")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


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
    """Collate function for multiple-choice batches."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def evaluate(model, dataloader, device):
    """Evaluate model on multiple-choice task."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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


def run_dapt_mlm(args, device):
    """
    Phase 1: Domain-Adaptive Pre-Training using Masked Language Modeling.
    Continues pre-training RoBERTa on unlabeled legal text.
    """
    print("\n" + "="*60)
    print("PHASE 1: DAPT - Masked Language Modeling")
    print("="*60)
    
    # Load tokenizer and MLM model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    
    # Load unlabeled corpus
    print(f"\nLoading unlabeled corpus: {args.unlabeled_corpus}")
    dataset = UnlabeledTextDataset(args.unlabeled_corpus, tokenizer, args.max_length)
    
    # Data collator for MLM (handles random masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Mask 15% of tokens
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.dapt_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.dapt_learning_rate)
    total_steps = len(dataloader) * args.dapt_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"\nMLM Training Config:")
    print(f"  Unlabeled examples: {len(dataset)}")
    print(f"  Epochs: {args.dapt_epochs}")
    print(f"  Batch size: {args.dapt_batch_size}")
    print(f"  Learning rate: {args.dapt_learning_rate}")
    
    start_time = time.time()
    dapt_stats = []
    
    for epoch in range(args.dapt_epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        
        progress_bar = tqdm(dataloader, desc=f"DAPT Epoch {epoch+1}/{args.dapt_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        print(f"\nDAPT Epoch {epoch+1}: MLM Loss={avg_loss:.4f}, Time={epoch_time/60:.2f}min")
        
        dapt_stats.append({
            'epoch': epoch + 1,
            'mlm_loss': avg_loss,
            'time_minutes': epoch_time / 60
        })
    
    dapt_time = time.time() - start_time
    
    # Save the domain-adapted model
    dapt_model_dir = os.path.join(args.output_dir, 'dapt_model')
    os.makedirs(dapt_model_dir, exist_ok=True)
    model.save_pretrained(dapt_model_dir)
    tokenizer.save_pretrained(dapt_model_dir)
    
    print(f"\n✓ DAPT completed in {dapt_time/60:.2f} minutes")
    print(f"✓ Domain-adapted model saved to {dapt_model_dir}")
    
    return dapt_model_dir, dapt_time, dapt_stats


def run_fine_tuning(dapt_model_dir, args, device):
    """
    Phase 2: Fine-tuning the domain-adapted model on CaseHOLD.
    """
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning on CaseHOLD")
    print("="*60)
    
    # Load tokenizer and model from DAPT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(dapt_model_dir)
    model = AutoModelForMultipleChoice.from_pretrained(
        dapt_model_dir,
        ignore_mismatched_sizes=True  # Classification head will be randomly initialized
    )
    model.to(device)
    
    # Load datasets
    train_dataset = CaseHOLDDataset(args.train_data, tokenizer, args.max_length)
    val_dataset = CaseHOLDDataset(args.val_data, tokenizer, args.max_length)
    test_dataset = CaseHOLDDataset(args.test_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.ft_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.ft_batch_size, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.ft_batch_size, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.ft_learning_rate)
    total_steps = len(train_loader) * args.ft_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    print(f"\nFT Config: {args.ft_epochs} epochs, batch={args.ft_batch_size}, lr={args.ft_learning_rate}")
    
    start_time = time.time()
    best_val_acc = 0
    ft_stats = []
    
    for epoch in range(args.ft_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{args.ft_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}: Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        ft_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1']
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            ft_model_dir = os.path.join(args.output_dir, 'final_model')
            os.makedirs(ft_model_dir, exist_ok=True)
            model.save_pretrained(ft_model_dir)
            tokenizer.save_pretrained(ft_model_dir)
    
    # Final test evaluation
    test_metrics = evaluate(model, test_loader, device)
    ft_time = time.time() - start_time
    
    return test_metrics, ft_time, ft_stats, best_val_acc


def main(args):
    """Main function orchestrating DAPT pipeline."""
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    total_start = time.time()
    
    # Phase 1: Domain-Adaptive Pre-Training
    dapt_model_dir, dapt_time, dapt_stats = run_dapt_mlm(args, device)
    
    # Phase 2: Fine-tuning on downstream task
    test_metrics, ft_time, ft_stats, best_val_acc = run_fine_tuning(dapt_model_dir, args, device)
    
    total_time = time.time() - total_start
    
    # Count unlabeled documents
    with open(args.unlabeled_corpus, 'r') as f:
        unlabeled_size = len([t for t in f.read().split('\n\n') if t.strip()])
    
    # Save comprehensive results
    results = {
        'method': 'dapt_ft',
        'model_name': args.model_name,
        'unlabeled_corpus': args.unlabeled_corpus,
        'unlabeled_size': unlabeled_size,
        'dapt_epochs': args.dapt_epochs,
        'dapt_batch_size': args.dapt_batch_size,
        'dapt_learning_rate': args.dapt_learning_rate,
        'ft_epochs': args.ft_epochs,
        'ft_batch_size': args.ft_batch_size,
        'ft_learning_rate': args.ft_learning_rate,
        'seed': args.seed,
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'dapt_time_minutes': dapt_time / 60,
        'ft_time_minutes': ft_time / 60,
        'total_time_hours': total_time / 3600,
        'gpu_hours_estimate': total_time / 3600,
        'dapt_stats': dapt_stats,
        'ft_stats': ft_stats
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DAPT PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Unlabeled corpus size: {unlabeled_size} documents")
    print(f"DAPT time: {dapt_time/60:.2f} minutes")
    print(f"Fine-tuning time: {ft_time/60:.2f} minutes")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"\n✓ Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPT + Fine-tuning for CaseHOLD")
    
    # Model and data paths
    parser.add_argument('--model_name', default='roberta-base', type=str,
                        help='Pretrained model name')
    parser.add_argument('--unlabeled_corpus', required=True, type=str,
                        help='Path to unlabeled legal corpus for DAPT')
    parser.add_argument('--train_data', default='data/casehold/train.jsonl', type=str,
                        help='Path to training data')
    parser.add_argument('--val_data', default='data/casehold/val.jsonl', type=str,
                        help='Path to validation data')
    parser.add_argument('--test_data', default='data/casehold/test.jsonl', type=str,
                        help='Path to test data')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Output directory for models and results')
    
    # DAPT (Phase 1) hyperparameters
    parser.add_argument('--dapt_epochs', default=1, type=int,
                        help='Number of epochs for DAPT MLM')
    parser.add_argument('--dapt_batch_size', default=8, type=int,
                        help='Batch size for DAPT')
    parser.add_argument('--dapt_learning_rate', default=5e-5, type=float,
                        help='Learning rate for DAPT')
    
    # Fine-tuning (Phase 2) hyperparameters
    parser.add_argument('--ft_epochs', default=3, type=int,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--ft_batch_size', default=8, type=int,
                        help='Batch size for fine-tuning')
    parser.add_argument('--ft_learning_rate', default=2e-5, type=float,
                        help='Learning rate for fine-tuning')
    
    # General settings
    parser.add_argument('--max_length', default=256, type=int,
                        help='Maximum sequence length')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
