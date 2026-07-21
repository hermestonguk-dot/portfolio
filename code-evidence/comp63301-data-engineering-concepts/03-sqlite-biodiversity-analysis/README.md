# Lab 03: SQLite Biodiversity Analysis

## What This Shows

This lab uses Pandas and SQLite to analyse Sites of Biological Importance data. The code imports tabular data, builds a SQLite database, normalises feature codes, and writes query outputs to CSV.

## Contents

- `comp63301_cw6.py`: completed submitted Python script.
- `comp63301_cw6_original_notebook.ipynb`: original notebook-style coursework material for context.
- `gmeu_sbi_tabular.xlsx`: local Excel dataset so the script can run even if the public download is unavailable.
- `requirements.txt`: Python packages for local execution.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the submitted script:

```bash
python comp63301_cw6.py
```

The script writes outputs such as `districts.csv`, `hectars.csv`, `features.csv`, `featurecodes.csv`, `sitefeatures.csv`, and other query results. It also attempts to download the Excel source file; the same Excel file is included locally as a fallback.

## Skills Demonstrated

- Pandas-to-SQLite ingestion
- SQL filtering and aggregation
- regex parsing of feature codes
- table normalisation
- CSV result generation

