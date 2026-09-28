import csv
import math
import sys
import time
from collections import Counter

import requests

from implement_alg1 import (
    DBO_PREFIX,
    DBPEDIA_SPARQL_URL,
    Cell_Sampling,
    Class_Querying,
    Entity_Lookup,
    read_column,
    read_targets,
    write_results,
)


RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
COVERAGE_THRESHOLD = 0.8


def Query_Supported_Classes(classes):
    """
    Return each direct class plus its DBpedia ontology superclasses. These are
    the classes supported by one matched entity.
    """
    unique_classes = sorted(set(classes))
    if not unique_classes:
        return set()

    values = " ".join(f"<{label}>" for label in unique_classes)
    query = f"""
    SELECT DISTINCT ?supported
    WHERE {{
      VALUES ?direct {{ {values} }}
      ?direct <{RDFS_SUBCLASS_OF}>* ?supported .
      FILTER(STRSTARTS(STR(?supported), "{DBO_PREFIX}"))
    }}
    """
    params = {"query": query, "format": "json"}
    headers = {"Accept": "application/sparql-results+json"}

    try:
        response = requests.get(DBPEDIA_SPARQL_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return set(unique_classes)

    data = response.json()
    return {
        result["supported"]["value"]
        for result in data["results"]["bindings"]
    }


def Query_Class_Depths(classes):
    """
    Approximate specificity by counting DBpedia ontology ancestors. Deeper
    classes are treated as more specific.
    """
    unique_classes = sorted(set(classes))
    if not unique_classes:
        return {}

    values = " ".join(f"<{label}>" for label in unique_classes)
    query = f"""
    SELECT ?class (COUNT(DISTINCT ?ancestor) AS ?depth)
    WHERE {{
      VALUES ?class {{ {values} }}
      OPTIONAL {{
        ?class <{RDFS_SUBCLASS_OF}>+ ?ancestor .
        FILTER(STRSTARTS(STR(?ancestor), "{DBO_PREFIX}"))
      }}
    }}
    GROUP BY ?class
    """
    params = {"query": query, "format": "json"}
    headers = {"Accept": "application/sparql-results+json"}

    try:
        response = requests.get(DBPEDIA_SPARQL_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return {label: 0 for label in unique_classes}

    data = response.json()
    depths = {label: 0 for label in unique_classes}
    for result in data["results"]["bindings"]:
        label = result["class"]["value"]
        depths[label] = int(result["depth"]["value"])
    return depths


def Class_Selection_Coverage_Specificity(entity_classes):
    """
    Improve Step 4 with an ontology-aware coverage-specificity rule:
    keep classes with high entity coverage, then choose the deepest one.
    """
    if not entity_classes:
        return "NONE"

    supported_by_entity = []
    for classes in entity_classes:
        supported_classes = Query_Supported_Classes(classes)
        if supported_classes:
            supported_by_entity.append(supported_classes)

    if not supported_by_entity:
        return "NONE"

    support_counts = Counter()
    for supported_classes in supported_by_entity:
        for label in supported_classes:
            support_counts[label] += 1

    max_support = max(support_counts.values())
    minimum_support = math.ceil(max_support * COVERAGE_THRESHOLD)
    candidates = [
        label for label, support in support_counts.items()
        if support >= minimum_support
    ]
    depths = Query_Class_Depths(candidates)

    return max(
        candidates,
        key=lambda label: (depths.get(label, 0), support_counts[label], label),
    )


def annotate_column_alg2(table_name, column_id, k):
    column_values = read_column(table_name, column_id)
    sampled_cells = Cell_Sampling(column_values, k)

    entity_classes = []
    for cell in sampled_cells:
        entity = Entity_Lookup(cell)
        if entity is None:
            continue

        classes = Class_Querying(entity)
        if classes:
            entity_classes.append(classes)

    return Class_Selection_Coverage_Specificity(entity_classes)


def accuracy_against_gt(predictions):
    with open("GT.csv", "r", encoding="utf-8", newline="") as f:
        gt_rows = list(csv.reader(f))

    correct = 0
    for pred, gt in zip(predictions, gt_rows):
        if pred[2] == gt[2]:
            correct += 1
    return correct / len(gt_rows)


def main():
    k = 5
    output_file = sys.argv[1] if len(sys.argv) > 1 else "results_alg2.csv"
    start = time.time()

    predictions = []
    targets = read_targets("Targets.csv")
    for i, (table_name, column_id) in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] Annotating {table_name}, column {column_id}")
        annotation = annotate_column_alg2(table_name, column_id, k)
        predictions.append((table_name, str(column_id), annotation))

    write_results(output_file, predictions)

    running_time = (time.time() - start) / 60
    accuracy = accuracy_against_gt(predictions)
    print("\nResults written to: " + output_file)
    print(f"Running time: {running_time:.2f} minutes")
    print(f"Accuracy against GT.csv: {accuracy:.2f}")


if __name__ == "__main__":
    main()
