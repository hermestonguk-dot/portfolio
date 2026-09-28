# Lab 04: Snakemake Carbon Data Pipeline

## Contents

This lab implements a Snakemake ETL pipeline for the Global Carbon Project fossil CO2 emissions dataset. It downloads source data, validates it with metadata, cleans country-level records, and generates per-country/per-source result files.

## Contents

- `environment.yaml`: top-level Conda environment for Snakemake.
- `config/`: dataset configuration and source URLs.
- `workflow/Snakefile`: submitted workflow rules.
- `workflow/scripts/`: Python scripts for download and validation.
- `workflow/notebooks/`: notebook steps used by workflow rules.
- `workflow/envs/`: per-rule Conda environments.

## How To Run

Create and activate the Snakemake environment:

```bash
conda env create -f environment.yaml
conda activate snakemake-etl
```

Run from this lab folder:

```bash
snakemake --use-conda --cores 1
```




