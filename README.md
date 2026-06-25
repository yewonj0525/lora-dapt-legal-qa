# LoRA vs DAPT in Legal QA

> Systematic comparison of Low-Rank Adaptation (LoRA) and Domain-Adaptive Pre-Training (DAPT) for legal question answering on the CaseHOLD dataset.

---

## 📌 Project Overview

Fine-tuning large language models for specialized domains is resource-intensive. This project investigates two parameter-efficient alternatives — **LoRA** and **DAPT** — and identifies the break-even point where domain adaptation becomes worthwhile.

**Key questions:**
- At what corpus size does DAPT outperform LoRA?
- How does DAPT performance scale with unlabeled data size?
- Can LoRA match full fine-tuning with only 0.24% of parameters?

**Evaluated on:** CaseHOLD — a 5-way multiple-choice legal QA benchmark with 53K examples.

---

## 👩‍💻 My Contributions

This was a 2-person team project. I was responsible for the following:

### 1. Baseline Full Fine-tuning
- Fine-tuned RoBERTa-base on the full CaseHOLD training set
- Achieved **74.04% test accuracy** — serves as the upper-bound reference for all comparisons
- Ran with 2 random seeds (42, 123) to verify result stability

### 2. DAPT Experiments (4 corpus sizes)
- Ran Domain-Adaptive Pre-Training with unlabeled legal corpora of varying sizes
- Systematically compared performance across corpus sizes to identify the optimal pre-training scale

| Corpus Size | Test Accuracy | vs Baseline |
|---|---|---|
| DAPT 500 | 73.75% | -0.29% |
| **DAPT 1000** | **74.73%** | **+0.69%** |
| DAPT 3000 | 74.20% | +0.16% |
| DAPT 10000 | 73.96% | -0.08% |

**Key finding:** DAPT with ~1,000 documents outperforms full fine-tuning, but performance is non-monotonic — more unlabeled data doesn't always help.

---

## 📊 Key Results

| Method | Test Accuracy | GPU Hours | Parameters |
|---|---|---|---|
| Zero-shot | 14.23% | 0.00 | 0 |
| Baseline FT | 74.04% | 1.88 | 124.6M |
| LoRA FT | 70.13% | 1.40 | 0.3M (0.24%) |
| DAPT 500 | 73.75% | 1.79 | 124.6M |
| **DAPT 1000** | **74.73%** | 1.79 | 124.6M |
| DAPT 3000 | 74.20% | 1.81 | 124.6M |
| DAPT 10000 | 73.96% | 1.83 | 124.6M |

**Key findings:**
1. Break-even point (DAPT vs LoRA): < 500 documents
2. Optimal DAPT corpus size: ~1,000 documents
3. LoRA achieves 95% of baseline performance with only 0.24% of parameters

---

## 📁 Project Structure

```
├── scripts/
│   ├── prepare_data.py          # Convert CaseHOLD CSV to JSONL
│   ├── prepare_unlabeled.py     # Extract unlabeled corpus for DAPT
│   ├── eval_zeroshot.py         # Zero-shot evaluation
│   ├── train_baseline.py        # Standard full fine-tuning
│   ├── train_lora.py            # LoRA fine-tuning
│   ├── train_dapt.py            # DAPT + fine-tuning pipeline
│   ├── compare_results.py       # Results analysis and comparison
│   └── run_all_experiments.sh   # Run all experiments
├── requirements.txt
└── README.md
```

> **Note:** Raw data not included. Download the CaseHOLD dataset from the [official repository](https://github.com/reglab/casehold).

---

## ⚙️ Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and prepare data
```bash
git clone https://github.com/reglab/casehold.git

python scripts/prepare_data.py \
    --input_dir casehold/data/all \
    --output_dir data/casehold

python scripts/prepare_unlabeled.py \
    --input_csv casehold/data/all/train.csv \
    --output_dir data/unlabeled \
    --sizes 500 1000 3000 10000
```

### 3. Run experiments
```bash
# Run all experiments (parallel on 2 GPUs)
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh

# Or run individually
python scripts/train_baseline.py --output_dir results/baseline --epochs 3
python scripts/train_dapt.py --unlabeled_corpus data/unlabeled/legal_corpus_1000.txt --output_dir results/dapt_1000

# Compare results
python scripts/compare_results.py --results_dir results
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Models:** RoBERTa-base (HuggingFace Transformers)
- **Frameworks:** PyTorch, PEFT (LoRA)
- **Tools:** HuggingFace Datasets, scikit-learn, NumPy, Git
- **Hardware:** NVIDIA A100 GPU (40GB), 2 GPUs

---

## 👥 Team

| Name | Role |
|---|---|
| Yewon Joung | Baseline full fine-tuning, DAPT experiments (4 corpus sizes) |
| Grace Wang | — |

*UNC Chapel Hill — COMP 790-183: Transfer Learning (Fall 2025)*
