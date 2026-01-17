# Self-Supervised Learning for Gene Expression Data

This repository contains the full codebase to reproduce the experiments presented in the project  
**"Self-Supervised Learning for Gene Expression Data"**  
*(Master 2 GENIOMHE-AI, Université d’Évry Paris-Saclay)*.

The project investigates how **self-supervised learning (SSL)** with biologically informed objectives improves cancer type classification from high-dimensional RNA-seq data, especially in low-label regimes.

---

## 1. Project Overview

**Task:**  
Multi-class cancer type classification from gene expression (RNA-seq) data

**Domain:**  
Healthcare / Bioinformatics

**Key challenge:**  
~20,000 gene features with limited labeled samples

**Proposed solution:**  
Self-supervised pretraining on unlabeled data using biologically motivated objectives

---
#### Project description
* Document:available [here](https://github.com/ai4precision-medicine/2526-m2geniomhe-SSL-tcga/blob/main/description/Projet%20GENIOHME.pdf).
* Slides: available [here](https://github.com/ai4precision-medicine/2526-m2geniomhe-SSL-tcga/blob/main/description/project_proposal_DRADJAT-1.pdf)

## SSL Framework

The SSL framework combines:

- **Pathway activity prediction** (ssGSEA, KEGG)
- **Masked gene reconstruction** (MAE-style)
- **Contrastive representation learning** (SimCLR-style)


#### Data 
All data used in this project are derived from The Cancer Genome Atlas (TCGA): data are available [here](https://drive.google.com/drive/folders/1U4I8qabkJ_Oo6l1CmBLvju2uY54gCSdM?usp=sharing)

All data used in this project are derived from The Cancer Genome Atlas (TCGA).

- Data type: bulk RNA-seq gene expression
- Features: ~19,800 gene expression values per sample
- Task: multi-class cancer type classification

Dataset splits:
- Pretraining (unlabeled): ~7,300 samples
- Fine-tuning (labeled): 1,000 samples
- Test set: 1,000 samples

Labels correspond to cancer types (TCGA annotations)

####  Gene set enrichment score / Pathway score computation
Tutorial: use the [gseapy](https://gseapy.readthedocs.io/en/latest/gseapy_example.html#Single-Sample-GSEA-example) with [Mygene](https://docs.mygene.info/projects/mygene-py/en/latest/) package for the gene mapping.

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



Contact: kevin.dradjat@univ-evry.fr
