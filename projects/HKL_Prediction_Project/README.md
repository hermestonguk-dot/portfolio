# HKL Racing Prediction Project

This personal project is for predicting whether a Hong Kong horse racer is likely to finish in the top 4. 

## Project Structure

```text
.
|-- DataSets/
|   `-- Prediction/
|       |-- historical_races.csv
|       `-- new_upcoming_races.csv
|-- src/hkl_prediction/            # cleaned Python package
|-- tests/                         # focused unit tests
|-- RESULTS.md                     # published full-data evaluation
|-- outputs/                       # generated when the pipeline runs
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Pipeline

1. Loads historical race rows from `DataSets/Prediction/historical_races.csv`.
2. Creates a binary target: `HitTop4 = FinishPosition <= 4`.
3. Cleans numeric and categorical features.
4. Adds race-level features such as odds rank, rating rank, field size, and implied win probability.
5. Trains several scikit-learn models with a time-aware validation split.
6. Tunes a threshold per model on validation F1 score.
7. Records accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
8. Saves models and metadata to `outputs/models/`.
9. Generates a readable evaluation report at `outputs/evaluation_report.md`.
10. Scores `DataSets/Prediction/new_upcoming_races.csv`.
11. Writes ranked predictions to `outputs/predictions.csv` and `outputs/predictions.xlsx`.

The latest full-data results are published in [RESULTS.md](RESULTS.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

For runtime dependencies only:

```bash
pip install -r requirements.txt
```

## Run The Full Offline Pipeline

From this folder:

```bash
hkl-predict run
```

Equivalent module command:

```bash
python -m hkl_prediction run
```

This creates the trained models, predictions, raw metrics, and a Markdown evaluation under `outputs/`.

## Fast Smoke Test

Use a smaller recent sample to check the environment quickly:

```bash
hkl-predict run --sample-size 5000 --output-dir outputs_smoke
```

## Main Commands

Train models only:

```bash
hkl-predict train
```

Predict with existing models:

```bash
hkl-predict predict
```

Run train and predict:

```bash
hkl-predict run
```

Publish the saved evaluation as the root-level results page:

```bash
hkl-predict report --report-path RESULTS.md
```

Run the automated checks:

```bash
python -m unittest discover -s tests
```

Custom paths:

```bash
hkl-predict run ^
  --historical DataSets/Prediction/historical_races.csv ^
  --upcoming DataSets/Prediction/new_upcoming_races.csv ^
  --output-dir outputs
```


