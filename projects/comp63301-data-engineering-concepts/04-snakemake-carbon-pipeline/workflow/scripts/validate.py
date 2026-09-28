from snakemake.script import snakemake
from frictionless import Package, Resource, Schema, describe

from pathlib import Path
from shutil import copy

def validate_csv(csv, metadata, log, destination):
    resource = Resource(csv)
    resource.schema = Schema(metadata)
    report = resource.validate()
    with (log.open("w", encoding="utf-8") as fb):
        fb.write(report.to_json())
    if (report.valid):
        copy(csv, destination)

csv = Path(snakemake.input[0])
metadata = Path(snakemake.input[1])


log_obj = snakemake.log
if isinstance(log_obj, (list, tuple)):
    log = Path(log_obj[0])
else:
    log = Path(log_obj)

destination = Path(snakemake.output[0])

log.parent.mkdir(parents=True, exist_ok=True)
destination.parent.mkdir(parents=True, exist_ok=True)

validate_csv(csv, metadata, log, destination)