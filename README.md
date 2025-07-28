#  Identifying the Case Holding from a Citing Prompt  
### A Comparative Study of Language Models for Legal Text Classification

<p align="center">
  <img src="https://img.shields.io/badge/NLP-LegalBERT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Task-Multiple_Choice_Classification-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-CaseHOLD-orange?style=flat-square"/>
</p>

##  Overview

Legal professionals often spend significant time locating the exact *holding*—the single sentence that summarizes a court's decision—from cited legal cases. In this project, we use various NLP models to automate this process using the [CaseHOLD](https://huggingface.co/datasets/casehold/casehold) dataset.

We benchmark the performance of 5 models on a multiple-choice classification task where each citing prompt is associated with five candidate holdings, and the goal is to identify the correct one.

---

##  Problem Statement

> **Given:** A citing prompt + 5 candidate holdings  
> **Goal:** Predict the correct holding (0 to 4)  
>  
> This is framed as a single-sentence classification task with significant domain-specific language complexity.

---

##  Dataset

- **Source:** CaseHOLD dataset from Hugging Face
- **Format:**  
  - `citing_prompt`  
  - 5 `candidate_holdings`  
  - `label` (integer 0–4)
- **Split:**  
  - 5000 samples for training  
  - 1000 samples for validation  
  - 1000 samples for testing  

---

##  Models Compared

| Model         | Type               | Pretraining         | Highlights                              |
|---------------|--------------------|---------------------|------------------------------------------|
| **LegalBERT** | BERT-based         | Legal documents     | Domain-specific, excels in legal context |
| **DistilBERT**| Compressed BERT    | General domain      | Faster and lighter than BERT             |
| **GPT-2**     | Decoder Transformer| General domain      | Scores completions based on likelihood   |
| **T5**        | Text-to-Text       | General domain      | Generates correct answer as a letter     |
| **LSTM**      | Custom BiLSTM      | From scratch        | Baseline with embeddings + attention     |

---

##  Methodology

###  Preprocessing
- Tokenized using Hugging Face tokenizers
- Max token length:
  - **512** for transformer models
  - **128** for LSTM model (with custom vocab of 20k tokens)
- Multiple-choice format: 5 prompt+choice pairs per sample

### Training Setup
- **Optimizer:** Adam / AdamW
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 10
- **Batch Sizes:** 
  - 8 (transformer models), 
  - 16/4/1 for custom model

---

## Results

| Model         | Accuracy | Macro F1 | Weighted F1 |
|---------------|----------|----------|-------------|
| **T5**        | **77.6** | **77.3** | **77.4**     |
| LegalBERT     | 75.6     | 75.3     | 75.1         |
| DistilBERT    | 61.7     | 61.1     | 61.3         |
| GPT-2         | 53.8     | 53.5     | 53.8         |
| LSTM (custom) | 28.2     | 28.2     | 28.2         |

>  **T5 performed best**, likely due to its generative design.  
>  **LegalBERT** showed strong results due to legal-domain training.

---

