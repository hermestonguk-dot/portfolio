# DBpedia Column Annotation

This project annotates table columns with DBpedia ontology classes. The first
algorithm samples distinct cell values, looks up their top DBpedia entities,
collects their directly declared ontology classes, and selects the most common
class. The second algorithm expands each class through the ontology hierarchy
and chooses a class that balances coverage with specificity.

## Run

Install the only external dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The original coursework input layout is required:

```text
Targets.csv
GT.csv                         # required for Algorithm 2 accuracy output
tables/<table-name>.csv
```

Then run from this directory:

```powershell
python implement_alg1.py
python implement_alg2.py results_alg2.csv
```

The scripts query the public DBpedia Lookup and SPARQL endpoints.
