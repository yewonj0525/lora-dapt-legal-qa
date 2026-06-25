# LoRA vs DAPT in Legal QA

Systematic comparison of Low-Rank Adaptation (LoRA) and Domain-Adaptive Pre-Training (DAPT) for legal question answering on the CaseHOLD dataset.

---

## 👩‍💻 My Contributions

This was a 2-person team project. I was responsible for the following:

### 1. Baseline Full Fine-tuning
- Fine-tuned RoBERTa-base on the full CaseHOLD training set (standard full fine-tuning)
- Achieved **74.04% test accuracy** — serves as the upper-bound reference for comparison
- Ran with 2 random seeds (42, 123) to verify result stability

### 2. DAPT Experiments (4 corpus sizes)
- Ran Domain-Adaptive Pre-Training with unlabeled legal corpora of varying sizes
- Compared performance across corpus sizes to find the optimal pre-training scale

| Corpus Size | Test Accuracy | vs Baseline |
|---|---|---|
| DAPT 500 | 73.75% | -0.29% |
| **DAPT 1000** | **74.73%** | **+0.69%** |
| DAPT 3000 | 74.20% | +0.16% |
| DAPT 10000 | 73.96% | -0.08% |

**Key finding**: DAPT with ~1,000 documents outperforms full fine-tuning, but performance is non-monotonic — more data doesn't always help.

---

## Project Structure

```
project/
├── scripts/
│   ├── prepare_data.py         # Convert CaseHOLD CSV to JSONL
│   ├── prepare_unlabeled.py    # Extract unlabeled corpus for DAPT
│   ├── eval_zeroshot.py        # Zero-shot evaluation (baseline)
│   ├── train_baseline.py       # Standard full fine-tuning       ← Yewon
│   ├── train_lora.py           # LoRA fine-tuning
│   ├── train_dapt.py           # DAPT + fine-tuning pipeline     ← Yewon
│   ├── compare_results.py      # Results analysis and comparison
│   └── run_all_experiments.sh  # Run all experiments
├── data/
│   ├── casehold/               # Processed CaseHOLD data
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── unlabeled/              # Unlabeled corpora for DAPT
│       ├── legal_corpus_500.txt
│       ├── legal_corpus_1000.txt
│       ├── legal_corpus_3000.txt
│       └── legal_corpus_10000.txt
├── results/                    # Experiment results
│   ├── zeroshot/
│   ├── baseline/
│   ├── lora/
│   ├── dapt_500/
│   ├── dapt_1000/
│   ├── dapt_3000/
│   └── dapt_10000/
└── logs/                       # Training logs
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CaseHOLD dataset

```bash
git clone https://github.com/reglab/casehold.git
```

### 3. Prepare data

```bash
# Convert CSV to JSONL
python scripts/prepare_data.py \
    --input_dir casehold/data/all \
    --output_dir data/casehold

# Create unlabeled corpora for DAPT
python scripts/prepare_unlabeled.py \
    --input_csv casehold/data/all/train.csv \
    --output_dir data/unlabeled \
    --sizes 500 1000 3000 10000
```

---

## Running Experiments

### Run all experiments (parallel on 2 GPUs)

```bash
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

### Run individual experiments

```bash
# Zero-shot evaluation
python scripts/eval_zeroshot.py --output_dir results/zeroshot

# Standard fine-tuning (baseline)
python scripts/train_baseline.py \
    --output_dir results/baseline \
    --epochs 3 --batch_size 8

# LoRA fine-tuning
python scripts/train_lora.py \
    --output_dir results/lora \
    --lora_r 8 --lora_alpha 16 \
    --epochs 3 --batch_size 8

# DAPT + fine-tuning
python scripts/train_dapt.py \
    --unlabeled_corpus data/unlabeled/legal_corpus_1000.txt \
    --output_dir results/dapt_1000 \
    --dapt_epochs 1 --ft_epochs 3
```

### Compare results

```bash
python scripts/compare_results.py --results_dir results
```

---

## Key Results

| Method | Test Accuracy | GPU Hours | Parameters |
|--------|---------------|-----------|------------|
| Zero-shot | 14.23% | 0.00 | 0 |
| Baseline FT | 74.04% | 1.88 | 124.6M |
| LoRA FT | 70.13% | 1.40 | 0.3M (0.24%) |
| DAPT 500 | 73.75% | 1.79 | 124.6M |
| **DAPT 1000** | **74.73%** | 1.79 | 124.6M |
| DAPT 3000 | 74.20% | 1.81 | 124.6M |
| DAPT 10000 | 73.96% | 1.83 | 124.6M |

## Key Findings

1. **Break-even point (DAPT vs LoRA):** < 500 documents
2. **Optimal DAPT corpus size:** ~1,000 documents (non-monotonic scaling)
3. **LoRA efficiency:** 95% of baseline performance with 0.24% parameters

---

## Team

| Name | Contribution |
|---|---|
| Yewon Joung | Baseline full fine-tuning, DAPT experiments (4 corpus sizes) |
| Grace Wang | LoRA fine-tuning, results analysis |

*UNC Chapel Hill — COMP 790-183: Transfer Learning (Fall 2025)*

---

## References

[1] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.

[2] S. Gururangan, A. Marasovic, S. Swayamditta, K. Lo, I. Beltagy, D. Downey, and N. A. Smith. Don't stop pretraining: Adapt language models to domains and tasks. In *Proceedings of ACL*, 2020.

[3] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In *Proceedings of ICLR*, 2022.

[4] L. Zheng, X. Guo, J. Chen, K. Chalkidis, D. Ji, and A. Klavans. When does pretraining help? Assessing self-supervised learning for law and the CaseHOLD dataset. In *Proceedings of EMNLP*, 2021.
