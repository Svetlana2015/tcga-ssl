# Self-Supervised Learning for Gene Expression Data

## Abstract
Gene expression data are characterized by extremely high dimensionality (≈20,000 genes per sample) and a limited availability of labeled samples, which makes purely supervised learning approaches difficult to apply effectively. In this project, we investigate a self-supervised learning (SSL) framework for transcriptomic data that leverages biologically informed pathway activity profiles as auxiliary training signals.

We propose to use pathway profile prediction, computed via single-sample Gene Set Enrichment Analysis (ssGSEA), as a biologically motivated pretext task for self-supervised pretraining. This objective is combined with masked gene reconstruction and contrastive learning to encourage the model to learn structured and transferable representations from unlabeled RNA-seq data. The learned encoder is subsequently evaluated on a downstream cancer type classification task.

Experimental results demonstrate that self-supervised pretraining substantially improves classification performance compared to a fully supervised baseline, particularly in low-label regimes. These findings highlight the value of incorporating biological prior knowledge into self-supervised representation learning for high-dimensional transcriptomic data.

---

## Approach Overview (Pipeline Diagram)

**Self-supervised pretraining**


```text
RNA-seq sample (genes)
        |
        v
  Shared Gene Expression Encoder
        |
        v
   +----+---------------------------+---------------------------+
   |                                |                           |
   v                                v                           v
Pathway Activity Prediction   Masked Gene Reconstruction   Contrastive Learning
       (ssGSEA)                     (MAE-style)                 (NT-Xent)
```


After pretraining, the encoder is reused for downstream cancer classification via:
- frozen encoder (linear probing),
- or full fine-tuning with labeled data.

---

## Data Preparation

All data used in this project are derived from The Cancer Genome Atlas (TCGA): data are available [here](https://drive.google.com/drive/folders/1U4I8qabkJ_Oo6l1CmBLvju2uY54gCSdM?usp=sharing)

- Data type: bulk RNA-seq gene expression
- Features: ~19,800 gene expression values per sample
- Task: multi-class cancer type classification

Dataset splits:
- Pretraining (unlabeled): ~7,300 samples
- Fine-tuning (labeled): 1,000 samples
- Test set: 1,000 samples

Labels correspond to cancer types (TCGA annotations)

## Project Workflow and Experimental Steps

This project follows a complete and reproducible pipeline for self-supervised learning on high-dimensional transcriptomic data. All stages described below were implemented and experimentally evaluated.

### 1. Data Processing and Dataset Preparation

Raw RNA-seq gene expression data and clinical labels are preprocessed using a dedicated pipeline implemented in `preprocessing.py`. The goal of this stage is to construct clean, standardized, and reproducible datasets for self-supervised pretraining, supervised fine-tuning, and evaluation.

**Input data requirements:**
- A gene expression table in Parquet format containing a `caseID` column and gene expression features;
- A labels table in Parquet format containing `cases` and `cancer_type` columns.

The following preprocessing steps are applied:

- **Sample identifier alignment:**  
  Sample identifiers are extracted from the labels table (`cases` field) and matched with the `caseID` column in the expression matrix. Only samples present in both tables are retained.

- **Merge of expression and labels:**  
  Gene expression features and cancer type labels are merged into a single dataframe to ensure consistent sample alignment.

- **Feature selection:**  
  All columns except `caseID` and `cancer_type` are treated as gene expression features. No additional gene filtering is applied at this stage.

- **Stratified dataset splitting:**  
  The merged dataset is split into three non-overlapping subsets using stratified sampling on `cancer_type`:
  - **Test set:** fixed number of samples (default: 1,000),
  - **Fine-tuning set:** fixed number of samples (default: 1,000),
  - **Pretraining set:** remaining samples or a user-defined subset (default: ~7,000).
  
  Stratification ensures that class proportions are preserved across all splits.

- **Feature scaling:**  
  Gene expression features are standardized using `StandardScaler`.  
  Importantly, the scaler is **fitted only on the pretraining dataset** and then applied to the fine-tuning and test sets to avoid data leakage.

- **Output generation:**  
  The following datasets are saved in Parquet format:
  - `pretrain.parquet` (unlabeled, used for self-supervised learning),
  - `finetune.parquet` (labeled, used for supervised fine-tuning),
  - `test.parquet` (labeled, used exclusively for evaluation).
  
  Optionally, the full merged dataset (`full.parquet`) and unused samples (`unused.parquet`) are also saved for traceability.

This preprocessing pipeline guarantees reproducible data splits, consistent feature scaling, and strict separation between training and evaluation data.


---

### 2. Baseline Supervised Training
A fully supervised multilayer perceptron (MLP) classifier was implemented as a baseline reference model. The baseline:

- Uses raw gene expression vectors as input;
- Consists of two hidden layers with ReLU activations;
- Is trained exclusively on labeled data without any form of pretraining.

To evaluate sensitivity to label availability, the model was trained using increasing subsets of labeled data (from 100 to 1,000 samples). Stratified sampling was applied to preserve class distributions. Performance was evaluated on a held-out test set using classification accuracy, reported as mean ± standard deviation across multiple runs.

This baseline establishes a reference point for assessing the benefits of self-supervised pretraining.

---

### 3. Pathway Score Computation (ssGSEA Pipeline)
To introduce biological prior knowledge, pathway activity profiles were computed using single-sample Gene Set Enrichment Analysis (ssGSEA):

- KEGG pathway gene sets were obtained in GMT format;
- Gene expression matrices were transposed to match ssGSEA input requirements (genes × samples);
- ssGSEA was applied using rank-based normalization;
- Computation was performed in batches to ensure memory efficiency;
- Normalized Enrichment Scores (NES) were extracted as pathway activity values.

The resulting pathway score matrices (samples × pathways) were saved to disk and reused throughout the experiments to ensure reproducibility.

---

### 4. Self-Supervised Pretraining
A self-supervised learning (SSL) framework was implemented to learn representations from unlabeled RNA-seq data. A shared encoder network was pretrained using three complementary objectives:

- **Pathway profile prediction**: regression of ssGSEA pathway scores to introduce biologically informed supervision;
- **Masked gene reconstruction**: MAE-style reconstruction of randomly masked gene expression values to promote robustness;
- **Contrastive learning**: SimCLR-style contrastive objective applied to augmented views of the same sample.

All objectives share a single encoder and are optimized jointly using a weighted loss function. Pretraining was performed for multiple epochs on unlabeled data, and training convergence was monitored through loss curves.

---

### 5. Fine-Tuning Process
After self-supervised pretraining, the learned encoder was evaluated on a downstream cancer type classification task using two strategies:

- **Frozen encoder (linear probing)**: only a classifier head is trained on top of the pretrained encoder;
- **Unfrozen encoder (full fine-tuning)**: both encoder and classifier are updated using labeled data.

Experiments were conducted using different proportions of labeled training data (10% to 100%) to study data efficiency. Performance was evaluated on an independent test set, enabling direct comparison with the supervised baseline.

---

### 6. Visualization and Model Comparison
Visualization and exploratory analysis were used throughout the project to interpret both raw data and learned representations:

- PCA was applied to highly variable genes to assess global structure and intrinsic dimensionality;
- t-SNE and UMAP were used to visualize nonlinear structure and sample clustering;
- Learning curves were generated to compare baseline and SSL-based models across label regimes.

These analyses demonstrate that self-supervised pretraining improves representation quality, data efficiency, and robustness, particularly in low-label settings.











### Preprocessing steps

1. **Gene ID mapping**  
   Ensembl IDs (ENSG) are mapped to gene symbols using `mygene`.

   - unmapped genes are removed  
   - duplicate gene symbols are aggregated by mean  
   - the resulting feature space is consistent across datasets  

2. **Pathway profile computation**

   - KEGG gene sets are used  
   - pathway activity profiles are computed using ssGSEA  
   - computed scores are stored and reused for reproducibility  

Tutorial: use the [gseapy](https://gseapy.readthedocs.io/en/latest/gseapy_example.html#Single-Sample-GSEA-example) with [Mygene](https://docs.mygene.info/projects/mygene-py/en/latest/) package for the gene mapping. 
---

## Training and Evaluation Pipeline

### 1. Baseline supervised model

- MLP with two hidden layers (512, 256)
- trained on varying numbers of labeled samples (100 → 1000)
- evaluation metric: classification accuracy

### 2. Self-supervised pretraining

- encoder trained on unlabeled RNA-seq data
- three self-supervised objectives:
  - pathway profile prediction (MSE loss)
  - masked gene reconstruction
  - contrastive learning (NT-Xent)
- combined loss function:

```text
L = α · L_path + β · L_mae + γ · L_ctr

α = 1.0
β = 0.3
γ = 0.1
````

### 3. Fine-tuning

- frozen encoder (linear probing)
- unfrozen encoder (full fine-tuning)
- evaluation across different proportions of labeled data

---

## Results

### Key findings

- The supervised baseline struggles in low-label regimes.
- SSL-pretrained representations significantly outperform the baseline when limited labeled data are available.
- Even with a frozen encoder, SSL representations provide strong performance.
- Full fine-tuning yields the best results when sufficient labeled data are available.

These results demonstrate improved data efficiency and robustness obtained through self-supervised pretraining with biologically informed objectives.

---

## Analysis and Discussion

The experiments highlight the importance of representation learning for high-dimensional transcriptomic data.  
Pathway activity prediction provides a meaningful biological inductive bias by encouraging the encoder to capture coordinated gene programs rather than isolated gene effects.

The combination of pathway-level supervision, masked reconstruction, and contrastive learning enables the model to learn representations that are both biologically interpretable and transferable.  
The strong performance of the frozen encoder indicates that the learned representations generalize well and are not overly dependent on task-specific fine-tuning.

### Limitations

- Only bulk RNA-seq data are considered.
- Pathway definitions depend on curated databases (e.g., KEGG).
- Results are evaluated on a limited set of cancer types.


## Available Documents

All reference documents are available in the `Description/` folder:

- **Project proposal (course description):** `Description/Project_Proposal.pdf`
- **Course guidelines / initial project description:** `Description/Course_Guidelines.pdf`
- **Final project report:** `Description/Final_Report.pdf`
- **Poster:** Poster
---

## Authors

Svetlana Sannikova

**Master 2 GENIOMHE-AI**, Université d’Évry Paris-Saclay

## Contact

kevin.dradjat@univ-evry.fr
sannikovasvetlana777@gmail.com






















## Reproducibility Instructions

### Environment Setup
Recommended environment:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CPU or GPU supported

Main dependencies:


###  Steps 
- Data processing and cleaning
- Baseline training
- Pathway score computation (GSEA) pipeline
- Pre-training process
- Fine-tuning process
- Visualization and comparison

### Installation

```bash
git clone -b Svetlana---branch https://github.com/ai4precision-medicine/2526-m2geniomhe-SSL-tcga.git
cd 2526-m2geniomhe-SSL-tcga
python -m venv env
env\Scripts\activate   # On Linux / macOS:source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage
All the scripts used for pretraining and finetuning with the differents methods are available on the scripts folder.

Baseline:
Example command
``` 
python -m scripts.run_baseline --epochs 50 --n_repeats 5 --out results.npz
```

Making pathways:
Example command
``` 
python -m scripts.make_pathways --input data/test.parquet --gmt data/kegg.gmt --case_id_col caseID --name test
```

Pretraining:
Example command
```
python -m scripts.run_ssl_pretrain --genes data/results_mapped/pretrain_mapped.parquet --pathways data/pathways/pretrain_pathways.parquet --save_dir experiments/ssl_pretrain

```

Finetuning:
Example command

```
python -m scripts.run_ssl_finetune --finetune data/results_mapped/finetune_mapped.parquet --test data/results_mapped/test_mapped.parquet --weights experiments/ssl_pretrain/ssl_model.pth --config experiments/ssl_pretrain/config.json --unfreeze_encoder
````


## Installation

### 1) Clone repository

```bash
git clone https://github.com/Svetlana2015/tcga-ssl.git
cd tcga-ssl
```

### 2) Create environment (recommended)

**Conda**

```bash
conda create -n tcga-ssl python=3.10 -y
conda activate tcga-ssl
```

**or venv (Windows)**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run pipeline (example)

```bash
python scripts/make_pathways.py
python scripts/run_ssl_pretrain.py
python scripts/run_ssl_finetune.py
python scripts/run_baseline.py
```

python -m scripts.run_preprocessing --expr data\raw\mRNA_coding.omics.parquet --labels data\raw\label.parquet --out_dir data --test_n 1000 --finetune_n 1000 --pretrain_n 7000



## Notes

* The scripts expect input files in `data/` (e.g., `kegg.gmt`, `pretrain.parquet`, `finetune.parquet`, `test.parquet`).
* Outputs are written to `data/results_mapped/`, `data/pathways/`, and `results/`.






