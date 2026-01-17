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

## Dataset

All data used in this project are derived from The Cancer Genome Atlas (TCGA): data are available [here](https://drive.google.com/drive/folders/1U4I8qabkJ_Oo6l1CmBLvju2uY54gCSdM?usp=sharing)

- Data type: bulk RNA-seq gene expression
- Features: ~19,800 gene expression values per sample
- Task: multi-class cancer type classification

Dataset splits:
- Pretraining (unlabeled): ~7,300 samples
- Fine-tuning (labeled): 1,000 samples
- Test set: 1,000 samples

Labels correspond to cancer types (TCGA annotations)

---

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

### 2. Baseline Supervised Training

A fully supervised multilayer perceptron (MLP) classifier is implemented as a baseline reference model. This baseline relies exclusively on labeled gene expression data and does not use any form of pretraining. This baseline training procedure is implemented in `baseline.py` and executed via `run_baseline.py`.


**Model architecture:**
- Input: standardized gene expression features;
- Two fully connected hidden layers:
  - 512 units + ReLU activation,
  - 256 units + ReLU activation;
- Optional dropout regularization;
- Output layer with one unit per cancer class.

**Training procedure:**
- The model is trained on the labeled fine-tuning dataset (`finetune.parquet`);
- Cancer type labels are encoded using `LabelEncoder`;
- For each experiment, a subset of the training data is selected using stratified sampling to preserve class proportions;
- Training subset sizes range from 100 samples up to the full fine-tuning set, with increments of 100 samples;
- For each training size, multiple independent runs are performed with different random seeds.

**Feature scaling:**
- Gene expression features are standardized using `StandardScaler`;
- The scaler is fitted on the selected training subset and applied to the test set for each run.

**Optimization and evaluation:**
- Optimizer: Adam;
- Loss function: cross-entropy loss;
- Training duration: fixed number of epochs per run;
- Evaluation metric: classification accuracy computed on an independent test set.

For each training size, the mean accuracy and standard deviation across runs are reported. This baseline serves as a reference to quantify the benefits of self-supervised pretraining in subsequent experiments.

### 3. Gene Identifier Mapping (ENSG → Gene Symbol)

Raw RNA-seq gene expression matrices use Ensembl gene identifiers (ENSG format), whereas most biological pathway databases (e.g., KEGG) are defined at the gene symbol level. Therefore, a dedicated gene identifier mapping step is applied before pathway analysis.

The mapping procedure is implemented in `mapping.py` and consists of the following steps:

- **Detection of gene columns:**  
  Columns whose names start with `ENSG` are identified as gene expression features. All other columns (e.g., `cancer_type`) are treated as metadata and preserved.

- **Sample identifier handling:**  
  If a `caseID` column is present, it is moved to the dataframe index to ensure stable sample alignment across processing steps.

- **Removal of Ensembl version suffixes:**  
  Ensembl identifiers may include version suffixes (e.g., `ENSG00000141510.12`). These suffixes are removed to ensure compatibility with annotation databases.

- **Querying gene annotations:**  
  Gene identifiers are mapped to gene symbols using the **MyGeneInfo** annotation service (`mygene`), querying the `ensembl.gene` scope for human genes.

- **Handling unmapped genes:**  
  Gene identifiers that cannot be mapped to valid gene symbols are removed from the feature space.

- **Aggregation of duplicate mappings:**  
  When multiple Ensembl IDs map to the same gene symbol, the corresponding expression columns are aggregated by computing their mean.

- **Gene symbol normalization:**  
  All gene symbols are converted to uppercase and stripped of extra whitespace to ensure consistent matching with pathway resources.

After mapping, the resulting gene expression matrices contain a unified and biologically consistent set of gene symbols while preserving all metadata columns. The mapped datasets are saved to disk for downstream analysis.

Gene identifier mapping relies on the [MyGeneInfo](https://docs.mygene.info/projects/mygene-py/en/latest/) service, while pathway activity computation is performed using the [gseapy](https://gseapy.readthedocs.io/en/latest/gseapy_example.html#Single-Sample-GSEA-example) library (ssGSEA) with KEGG gene sets.


### 4. Pathway Score Computation (ssGSEA)

To incorporate biological prior knowledge into the learning process, pathway activity profiles are computed from gene expression data using single-sample Gene Set Enrichment Analysis (ssGSEA).

This step is implemented in `pathways.py` and orchestrated via `make_pathways.py`. The procedure is as follows:

- **Pathway gene set loading:**  
  KEGG pathway definitions are provided in GMT format and parsed into a dictionary mapping pathway names to gene symbol lists.

- **Preparation of ssGSEA input:**  
  The gene expression matrix is filtered to include only gene expression features. Metadata columns (e.g., `cancer_type`) are excluded from ssGSEA computation.

- **Matrix transposition:**  
  The expression matrix is transposed to match ssGSEA input requirements:
  - rows: genes  
  - columns: samples

- **Batch-wise ssGSEA computation:**  
  ssGSEA is applied in batches of samples to ensure memory efficiency. Rank-based normalization is used, and normalized enrichment scores (NES) are extracted.

- **Aggregation of pathway scores:**  
  For each batch, ssGSEA results are reshaped into a pathway-by-sample matrix. All batches are concatenated to form a complete pathway activity matrix.

- **Reattachment of metadata:**  
  After pathway computation, metadata columns (including `cancer_type`) are reattached to the pathway score matrix to enable downstream supervised experiments.

The final output is a pathway activity matrix with shape *(samples × pathways)*, where each value represents the normalized enrichment score of a biological pathway for a given sample. These pathway profiles serve as biologically informed targets during self-supervised pretraining.


### 5. Self-Supervised Pretraining

Self-supervised pretraining is performed to learn meaningful representations from unlabeled RNA-seq data by leveraging biologically informed and generic self-supervised objectives. This step is implemented in `ssl_pretrain.py` and executed via `run_ssl_pretrain.py`.

**Input data:**
- Gene expression matrix (samples × genes), after preprocessing and gene identifier mapping;
- Pathway activity matrix (samples × pathways), computed using ssGSEA.
Both matrices are aligned by sample identifiers to ensure consistent correspondence.

**Model architecture:**
- A shared MLP encoder with three hidden layers (1024 → 512 → latent dimension);
- Layer normalization applied to the latent representation;
- Three task-specific heads sharing the same encoder:
  - a pathway regression head,
  - a masked gene reconstruction (MAE-style) decoder,
  - a projection head for contrastive learning.

**Self-supervised objectives:**
- **Pathway profile prediction:**  
  The encoder is trained to regress ssGSEA pathway activity scores using mean squared error loss, providing biologically informed supervision.
- **Masked gene reconstruction:**  
  A fixed fraction of gene features is randomly masked, and the model learns to reconstruct only the masked values.
- **Contrastive learning:**  
  Two augmented views of each sample are generated using Gaussian noise and random feature dropout. A SimCLR-style NT-Xent loss encourages similar samples to have closer latent representations.

**Training procedure:**
- All three objectives are optimized jointly using a weighted sum of losses:

```text
L = α · L_path + β · L_mae + γ · L_ctr
````
- Default loss weights are α = 1.0, β = 0.3, γ = 0.1;
- Training is performed for a fixed number of epochs using the Adam optimizer;
- The final pretrained model weights and training configuration are saved to disk for reproducibility.

This pretraining stage produces an encoder that captures structured, biologically relevant patterns in transcriptomic data without using any class labels.

### 6. Fine-Tuning and Linear Probing

The quality of the self-supervised representations is evaluated on a downstream cancer type classification task through fine-tuning and linear probing. This step is implemented in `ssl_finetune.py` and executed via `run_ssl_finetune.py`.

**Setup:**
- A pretrained encoder is loaded from the self-supervised pretraining stage;
- A linear classification head is added on top of the encoder;
- Cancer type labels are encoded using a shared `LabelEncoder` fitted on the fine-tuning dataset.

**Two evaluation strategies are considered:**
- **Frozen encoder (linear probing):**  
  The encoder weights are fixed, and only the linear classifier is trained.
- **Unfrozen encoder (full fine-tuning):**  
  Both the encoder and classifier are updated during training.

**Experimental protocol:**
- Fine-tuning is performed using different proportions of labeled training data, ranging from 10% to 100%;
- For each proportion, multiple independent runs are conducted using stratified sampling to preserve class distributions;
- Models are trained for a fixed number of epochs using the Adam optimizer and cross-entropy loss;
- Performance is evaluated on a held-out test set using classification accuracy.

**Outputs:**
- Classification accuracies (mean and standard deviation) for each training proportion;
- Saved model checkpoints for each run and the best-performing model;
- CSV files summarizing fine-tuning results for frozen and unfrozen settings.

This evaluation protocol enables a direct comparison between supervised baselines and self-supervised representations, highlighting the benefits of SSL in low-label regimes.


### 7. Model Summary, Visualization, and Comparison

To support model interpretability and facilitate comparison between supervised and self-supervised approaches, dedicated utilities are provided for model inspection and result visualization. These steps are implemented in `summary.py` and `visualization.py`.

#### Model Architecture Summary
To document the complexity and structure of the implemented models, architectural summaries are generated using `torchsummary`:

- **Baseline MLP summary:**  
  The baseline classifier architecture is summarized by automatically inferring the input dimensionality and number of output classes from the fine-tuning dataset. This provides a layer-by-layer overview of the supervised reference model.

- **SSL encoder summary:**  
  After self-supervised pretraining, only the encoder component (encoder layers + layer normalization) is extracted and summarized. This allows direct inspection of the learned representation model that is reused for downstream tasks.

These summaries provide transparency regarding model depth, parameter counts, and architectural differences between supervised and self-supervised approaches.

#### Performance Visualization

Several visualization routines are used to analyze and compare model performance:

- **Baseline learning curve:**  
  Classification accuracy of the supervised MLP is plotted as a function of the training subset size, reporting mean performance and standard deviation across multiple runs.

- **SSL vs baseline comparison:**  
  Training curves for the baseline model, SSL with frozen encoder (linear probing), and SSL with unfrozen encoder (full fine-tuning) are plotted on a common axis using training proportions.

- **Variability visualization:**  
  Shaded regions corresponding to ±1 standard deviation are displayed where available to highlight variability across runs.

All plots can be displayed interactively or saved to disk for inclusion in the final report and scientific poster.

---

## Results

The main experimental results are summarized as follows:

- The purely supervised baseline model exhibits limited performance when trained on small labeled datasets.
- Self-supervised pretraining substantially improves classification accuracy in low-label regimes.
- SSL representations remain effective even when the encoder is frozen, indicating strong transferability.
- Full fine-tuning of the pretrained encoder yields the highest performance when sufficient labeled data are available.

These results demonstrate that biologically informed self-supervised learning improves data efficiency and robustness for cancer classification from high-dimensional RNA-seq data.

---

## Analysis and Discussion
  
Directly modeling gene-level expression vectors is challenging due to the large feature space and limited availability of labeled samples. In this context, pathway activity prediction introduces a biologically meaningful inductive bias that encourages the encoder to focus on coordinated gene programs rather than isolated gene-level variations.

By using pathway profiles as a self-supervised pretext task, the model is guided toward learning structured latent representations that reflect underlying biological processes. The combination of pathway-level supervision with masked gene reconstruction and contrastive learning further promotes robustness and invariance to noise, leading to transferable representations.

The strong performance observed in the frozen-encoder (linear probing) setting indicates that the learned representations generalize well and are not overly dependent on task-specific fine-tuning. This suggests that the self-supervised encoder captures reusable biological signals that are relevant across different downstream classification scenarios.

These findings support the use of biologically informed self-supervised learning as an effective strategy for improving data efficiency and generalization in transcriptomic analysis.

---

### Limitations

- The study is limited to bulk RNA-seq data and does not address single-cell transcriptomic variability.
- Pathway activity profiles depend on curated gene set databases (e.g., KEGG), which may not fully capture all biological processes.
- Experimental evaluation is restricted to a limited number of cancer types, which may constrain the generalizability of the conclusions.

---
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
.\.venv\Scripts\activate      # On Linux / macOS: source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run pipeline 

All scripts used for preprocessing, pretraining, fine-tuning, and evaluation are available in the scripts/ directory.

Preprocessing
Example command
```
python -m scripts.run_preprocessing --expr data\raw\mRNA_coding.omics.parquet --labels data\raw\label.parquet --out_dir data --test_n 1000 --finetune_n 1000 --pretrain_n 7000
```

Baseline:
Example command
``` 
python -m scripts.run_baseline --epochs 50 --n_repeats 5 --out results.npz
```

Making pathways:
Example command
``` 
python -m scripts.make_pathways --input data/pretrain.parquet --gmt data/kegg.gmt --case_id_col caseID --name pretrain 
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
### Notes

* The scripts expect input files in `data/` (e.g., `kegg.gmt`, `pretrain.parquet`, `finetune.parquet`, `test.parquet`).
* Outputs are written to `data/results_mapped/`, `data/pathways/`, and `results/`.

---

## Available Documents

All reference documents are available in the `Description/` folder:

- **Project proposal (course description):** `Description/Project_Proposal.pdf`
- **Course guidelines / initial project description:** `Description/Course_Guidelines.pdf`
- **Final project report:** `Description/Final_Report.pdf`
- **Poster:** Poster
---

## Author

Svetlana Sannikova

**Master 2 GENIOMHE**, Université d’Évry Paris-Saclay

## Contact

kevin.dradjat@univ-evry.fr

sannikovasvetlana777@gmail.com





