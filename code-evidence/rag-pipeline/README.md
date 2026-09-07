# Retrieval-Augmented Generation Pipeline

This MSc coursework project evaluates a question-answering pipeline over a
Mediterranean cuisine corpus. It compares three chunking strategies, dense
embedding retrieval, BM25, cross-encoder reranking, four prompt strategies,
and several answer metrics.

## Included evidence

- `rag-final-pipeline.ipynb`: the complete experiment notebook.
- `rag_benchmark.json`: 90 question and reference-answer pairs.
- `section-aware_chunks.jsonl`, `sliding_fixed_chunks.jsonl`, and
  `sliding_sentence_chunks.jsonl`: the three chunking outputs used by the
  notebook.

The notebook evaluated 84 valid configurations. The best recorded setup was
section-aware chunks with enriched MPNet embeddings, cosine retrieval, and a
guided prompt. It achieved ROUGE-L **0.342**, compared with **0.263** for the
recorded baseline, an improvement of about **30%**. The comparison also
recorded Recall@5 **0.878**, MRR **0.742**, token F1 **0.315**, and BERTScore
**0.815** over 90 questions.

## Run it

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook rag-final-pipeline.ipynb
```

The notebook downloads the embedding, reranking, and language models from
Hugging Face. Full generation is computationally expensive; the notebook
contains a marked generation block and a smaller live-demo path.

The scores use lexical overlap and embedding similarity metrics. They show
how the tested components compare on this benchmark and do not establish that
the system is reliable for questions outside the corpus.
