- MPS is used when available otherwise CPU is used.
- FAISS uses one CPU thread to avoid the macOS crash
- Existing output files are overwritten when rerunning.
Create and activate a separate Conda environment with Python 3.12:

## Environment setup

```bash
conda create -n qa-llm-hw2 python=3.12 -y
conda activate qa-llm-hw2
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Create the output directory:

```bash
mkdir -p runs
```

## Generate the main retrieval runs

Ensure `data/question_bank.jsonl` is available.

Run the following to get the wiki-corpus:
```bash
mkdir -p data
python src/export_corpus.py
```

Run the following commands sequentially:

```bash
python src/run.py --method tfidf \
  --questions data/question_bank.jsonl \
  --articles data/wiki-corpus-v1.jsonl \
  --out runs/tfidf.jsonl

python src/run.py --method dpr \
  --questions data/question_bank.jsonl \
  --articles data/wiki-corpus-v1.jsonl \
  --out runs/dpr.jsonl

python src/run.py --method hybrid \
  --questions data/question_bank.jsonl \
  --articles data/wiki-corpus-v1.jsonl \
  --out runs/hybrid_0.5.jsonl

python src/run.py --method rrf \
  --questions data/question_bank.jsonl \
  --articles data/wiki-corpus-v1.jsonl \
  --out runs/rrf.jsonl
```


## Validate and evaluate

```bash
python eval.py data/question_bank.jsonl runs/tfidf.jsonl --validate
python eval.py data/question_bank.jsonl runs/dpr.jsonl --validate
python eval.py data/question_bank.jsonl runs/hybrid_0.5.jsonl --validate
python eval.py data/question_bank.jsonl runs/rrf.jsonl --validate


python eval.py data/question_bank.jsonl runs/tfidf.jsonl
python eval.py data/question_bank.jsonl runs/dpr.jsonl
python eval.py data/question_bank.jsonl runs/hybrid_0.5.jsonl
python eval.py data/question_bank.jsonl runs/rrf.jsonl
```
## Generate mini runs and run tests

```bash
python src/run.py --method tfidf \
  --questions tests/fixtures/mini_questions.jsonl \
  --articles tests/fixtures/mini_wiki.jsonl \
  --out runs/mini_sparse.jsonl

python src/run.py --method dpr \
  --questions tests/fixtures/mini_questions.jsonl \
  --articles tests/fixtures/mini_wiki.jsonl \
  --out runs/mini_dense.jsonl

python -m pytest tests -q
```