from snakemake.script import snakemake

import requests
from pathlib import Path

def download(url, destination):
    print("Downloading", url)
    r = requests.get(url)
    r.raise_for_status()    
    with Path(destination).open("wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)

# YOUR CODE GOES HERE
urls = snakemake.params["urls"]
outputs = snakemake.output

for url, dest in zip(urls, outputs):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    download(url, dest)