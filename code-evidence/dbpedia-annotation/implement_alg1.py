import csv
import random
import time
from collections import Counter

import requests



DBO_PREFIX = "http://dbpedia.org/ontology/"
DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"
DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"


def read_targets(filename):
    """
    Read in the target csv
    """
    targets = []
    with open(filename, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            targets.append((row[0], int(row[1])))
    return targets


def read_column(table_name, column_id):
    """
    Read the specific column from a table
    """
    values = []
    with open("tables/" + table_name + ".csv", "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            value = row[column_id].strip()
            if value:
                values.append(value)
    return values


def Cell_Sampling(column_values, k):
    """
    Step 1:Randomly select k non-empty, different cells from the column
    """
    different_cells = []
    for value in column_values:
        if value not in different_cells:
            different_cells.append(value)

    # If there are fewer than k different cells, use all available cells.
    if len(different_cells) <= k:
        return different_cells
    return random.sample(different_cells, k)


def Entity_Lookup(cell):
    """
    Step 2: For each cell c, lookup its top-1 corresponding entity using
    DBpedia lookup API (https://lookup.dbpedia.org/api/search?query=c). Thus k entities are
    collected, denoted as E.
    """
    params = {"query": cell, "maxResults": 1, "format": "JSON"}
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(DBPEDIA_LOOKUP_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()

    results = data.get("docs") or data.get("results") or []
    if not results:
        return None

    # Extract the URI of the top-1 result
    uri = results[0].get("resource") or results[0].get("uri") or results[0].get("id")
    if isinstance(uri, list):
        uri = uri[0]
    if not uri:
        return None

    if not uri.startswith("http"):
        uri = "http://dbpedia.org/resource/" + uri
    return uri.replace("https://dbpedia.org/resource/", "http://dbpedia.org/resource/")


def Class_Querying(entity):
    """
    Step 3: For each entity e in E, get its directly declared DBpedia ontology
    (starting with dbo:) class via a SPARQL query. All the matched entities lead to a list of
    classes, denoted as C.
    """

    query = f"""
    SELECT DISTINCT ?class
    WHERE {{
      <{entity}> a ?class .
      FILTER(STRSTARTS(STR(?class), "{DBO_PREFIX}"))
    }}
    """
    params = {"query": query, "format": "json"}
    headers = {"Accept": "application/sparql-results+json"}
    try:
        response = requests.get(DBPEDIA_SPARQL_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return []

    data = response.json()
    classes = []
    for result in data["results"]["bindings"]:
        classes.append(result["class"]["value"])
    return classes


def Class_Selection(classes):
    """
    Step 4: Select the most frequent class in C as the output annotation.
    """
    return Counter(classes).most_common(1)[0][0]


def annotate_column(table_name, column_id, k):
    """
    Runs all four steps of Alg 1 for a single target column. 
    """

    column_values = read_column(table_name, column_id)
    sampled_cells = Cell_Sampling(column_values, k)

    # Step 2 is applied to every sampled cell.
    entities = []
    for cell in sampled_cells:
        entity = Entity_Lookup(cell)
        if entity is not None:
            entities.append(entity)

    # Step 3 is applied to every matched entity.
    classes = []
    for entity in entities:
        classes.extend(Class_Querying(entity))

    return Class_Selection(classes)


def write_results(filename, rows):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows(rows)



def main():
    # Task 5.1 asks to run Alg 1 with k=5 over Targets.csv.
    k = 5
    output_file = "results_alg1.csv"
    start = time.time()

    # Annotate every target column listed in Targets.csv
    predictions = []
    targets = read_targets("Targets.csv")
    for i, (table_name, column_id) in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] Annotating {table_name}, column {column_id}")
        annotation = annotate_column(table_name, column_id, k)
        predictions.append((table_name, str(column_id), annotation))

    write_results(output_file, predictions)

    running_time = (time.time() - start) / 60

    print("\nResults written to: " + output_file)
    print(f"Running time: {running_time:.2f} minutes")


if __name__ == "__main__":
    main()
