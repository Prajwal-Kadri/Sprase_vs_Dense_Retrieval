"""Do not change the flags.

We will be calling this on a hidden question file that has only id, question and type.
Do not read gold titles, contexts or answers even if a file happens to contain them.
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
import json, shutil
import numpy as np
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)
import torch
import gc

import argparse
from pathlib import Path
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def chunking(step: int, text: str)-> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i<len(words):
        chunks.append(" ".join(words[i:i+step]))
        if i+step>=len(words):
            break
        i+=step
    return chunks

def top_k(scores, k):
    ranked_ids = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    top_ids = ranked_ids[:k]
    top_scores = [scores[i] for i in top_ids]
    return top_ids,top_scores

def normalize(chunk_ids, scores):
    low = min(scores)
    high = max(scores)
    if high==low:
        return {int(chunk_id): 0.0 for chunk_id in chunk_ids}
    return {
        int(chunk_id): float((score-low)/(high-low))
        for chunk_id, score in zip(chunk_ids, scores)
    }
def hybrid_rank(sparse_ids, sparse_scores, dense_ids, dense_scores, alpha):
    sparse_normalized = normalize(sparse_ids, sparse_scores)
    dense_normalized = normalize(dense_ids, dense_scores)

    candidate_ids = set(sparse_normalized) | set(dense_normalized)

    fused = []
    for chunk_id in candidate_ids:
        score = (
            alpha * dense_normalized.get(chunk_id, 0.0)
            + (1 - alpha) * sparse_normalized.get(chunk_id, 0.0)
        )
        fused.append((chunk_id, score))

    fused.sort(key=lambda pair: pair[1], reverse=True)

    return [chunk_id for chunk_id, score in fused]

def rrf_rank(sparse_ids, dense_ids, k=60):
    rrf_scores = {}

    for rank, chunk_id in enumerate(sparse_ids, start=1):
        chunk_id = int(chunk_id)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)

    for rank, chunk_id in enumerate(dense_ids, start=1):
        chunk_id = int(chunk_id)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (k + rank)

    ranked = sorted(
        rrf_scores.items(),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return [chunk_id for chunk_id, score in ranked]
def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
    
def sparse_encoding(question, vectorizer, X):
    q  = vectorizer.transform([question])
    return (X @ q.T).toarray().ravel()

def dense_encoding(question, index, config,question_tokenizer,question_encoder):
               
    question_inputs = question_tokenizer(
        [question],
        padding = True,
        truncation=True,
        return_tensors="pt",
    )
    question_inputs = {
        name: value.to(DEVICE)
        for name, value in question_inputs.items()
    }
    with torch.no_grad():
        question_vector = question_encoder(**question_inputs).pooler_output
    question_vector = question_vector.cpu().numpy().astype("float32")
    scores, indices = index.search(question_vector,100)
    return scores[0], indices[0]

def encode_questions_in_batches(question_texts, question_tokenizer, question_encoder, batch_size=32):
    all_vectors = []

    for start in range(0, len(question_texts), batch_size):
        end = start + batch_size
        batch_questions = question_texts[start:end]

        question_inputs = question_tokenizer(
            batch_questions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        question_inputs = {
            name: value.to(DEVICE)
            for name, value in question_inputs.items()
        }

        with torch.no_grad():
            vectors = question_encoder(**question_inputs).pooler_output

        all_vectors.append(vectors.cpu())

    return torch.cat(all_vectors, dim=0).numpy().astype("float32")

# "tfidf": {
#     "lowercase": true,
#     "stop_words": "english",
#     "ngram_range": [1, 1],
#     "sublinear_tf": false
#   },
def tf_idf(corpus,questions,config, chunk_set, corpus_chunk,out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        tfidf_config = config["tfidf"]
        vectorizer = TfidfVectorizer(stop_words=tfidf_config['stop_words'],lowercase=tfidf_config['lowercase'],ngram_range= tuple(tfidf_config['ngram_range']),sublinear_tf= tfidf_config['sublinear_tf'])
        X = vectorizer.fit_transform(corpus_chunk)
        for question in questions:
            seen_titles = set()
            titles = []
            scores = sparse_encoding(question["question"],vectorizer, X)
            ranked = scores.argsort()[::-1]
            sorted_chunk = [chunk_set[index] for index in ranked]
            for i in range(0,len(sorted_chunk)):
                if sorted_chunk[i]['title'] not in seen_titles:
                    seen_titles.add(sorted_chunk[i]['title'])
                    titles.append(sorted_chunk[i]['title'])
            row = {
                "id": question['id'],
                "titles": titles,
            }
            f.write(json.dumps(row)+"\n")
def dpr(corpus,questions,config, chunk_set, corpus_chunk,out_path):
    dpr_config = config["dpr"]
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_config["context_encoder"])
    context_encoder = DPRContextEncoder.from_pretrained(dpr_config["context_encoder"])
    context_encoder.to(DEVICE)
    context_encoder.eval()
    passages_chunk = [chunk["text"] for chunk in chunk_set]
    titles_chunk = [chunk["title"] for chunk in chunk_set]
    context_vectors = encode_context_in_batchees(titles_chunk,passages_chunk, context_tokenizer, context_encoder)
    del context_encoder
    del context_tokenizer
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    dpr_config = config["dpr"]
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        dpr_config["question_encoder"]
    )

    print("loading question model", flush=True)
    question_encoder = DPRQuestionEncoder.from_pretrained(
        dpr_config["question_encoder"]
    )
    print("question model loaded", flush=True)
    question_encoder.to(DEVICE)
    print("question model moved to device", flush=True)
    question_encoder.eval()

    question_texts = [question["question"] for question in questions]
    question_vectors = encode_questions_in_batches(
        question_texts, question_tokenizer, question_encoder
    )

    del question_encoder
    del question_tokenizer
    gc.collect()

    # Import FAISS only after all DPR/PyTorch encoding is complete.
    import faiss

    # Avoid the macOS OpenMP worker crash when FAISS and PyTorch coexist.
    faiss.omp_set_num_threads(1)

    index = faiss.IndexFlatIP(context_vectors.shape[1])
    index.add(context_vectors)
    dense_scores, dense_ids = index.search(question_vectors, 100)

    del context_vectors
    del question_vectors
    gc.collect()
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for question, ranked in zip(questions, dense_ids):
            seen_titles = set()
            titles = []
            for i in ranked:
                if i < 0:
                    continue
                title = chunk_set[int(i)]['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    titles.append(title)
            row = {
                "id": question['id'],
                "titles": titles,
            }
            f.write(json.dumps(row)+"\n")

def encode_context_in_batchees(titles_chunk, passages_chunk, context_tokenizer, context_encoder, batch_size=32):
    all_vectors = []

    for start in range(0, len(passages_chunk), batch_size):
        end = start + batch_size

        batch_titles = titles_chunk[start:end]
        batch_passages = passages_chunk[start:end]

        inputs = context_tokenizer(
            batch_titles,
            batch_passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(DEVICE) for name, value in inputs.items()}

        with torch.no_grad():
            vectors = context_encoder(**inputs).pooler_output

        all_vectors.append(vectors.cpu())

    return torch.cat(all_vectors, dim=0).numpy().astype("float32")

def run(method: str, questions_path: Path, articles_path: Path, out_path: Path) -> None:
    """Write a run file for one retriever.

    method is one of: tfidf, dpr, hybrid, rrf
    questions_path is JSONL with at least id and question
    articles_path is JSONL with title and text (Hugging Face wiki schema)
    out_path is JSONL of {"id": ..., "titles": [...]} already collapsed to articles
    """
    corpus = read_jsonl(articles_path)
    questions = read_jsonl(questions_path)
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root /"config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    
    chunk_set = []
    for article in corpus:
        for chunk in chunking(int(config["chunk_words"]),article["text"]):
            chunk_set.append({
                "title": article['title'],
                "text": chunk,
            })
    corpus_chunk = [chunk["text"] for chunk in chunk_set]
    if method == 'tfidf':
        tf_idf(corpus,questions,config, chunk_set, corpus_chunk,out_path)
          
    if method == "dpr":
        dpr(corpus,questions,config, chunk_set, corpus_chunk,out_path)
    if method == "hybrid" or method == "rrf":
        candidate_k = config["candidate_k"]
        dpr_config = config["dpr"]
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_config["context_encoder"])
        context_encoder = DPRContextEncoder.from_pretrained(dpr_config["context_encoder"])
        context_encoder.to(DEVICE)
        context_encoder.eval()
        passages_chunk = [chunk["text"] for chunk in chunk_set]
        titles_chunk = [chunk["title"] for chunk in chunk_set]
        context_vectors = encode_context_in_batchees(titles_chunk, passages_chunk, context_tokenizer, context_encoder)
        del context_encoder
        del context_tokenizer
        gc.collect()

        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_config["question_encoder"])
        question_encoder = DPRQuestionEncoder.from_pretrained(dpr_config["question_encoder"])
        question_encoder.to(DEVICE)
        question_encoder.eval()
        question_texts = [question["question"] for question in questions]
        question_vectors = encode_questions_in_batches(
            question_texts, question_tokenizer, question_encoder
        )
        del question_encoder
        del question_tokenizer
        gc.collect()

        # Import FAISS only after all DPR/PyTorch encoding is complete.
        import faiss
        # Avoid the macOS OpenMP worker crash when FAISS and PyTorch coexist.
        faiss.omp_set_num_threads(1)
        index = faiss.IndexFlatIP(context_vectors.shape[1])
        index.add(context_vectors)
        dense_scores, dense_ids = index.search(question_vectors, 100)
        del context_vectors
        del question_vectors
        gc.collect()

        tfidf_config = config["tfidf"]
        vectorizer = TfidfVectorizer(stop_words=tfidf_config['stop_words'],lowercase=tfidf_config['lowercase'],ngram_range= tuple(tfidf_config['ngram_range']),sublinear_tf= tfidf_config['sublinear_tf'])
        X = vectorizer.fit_transform(corpus_chunk)
        with open(out_path, "w", encoding = 'utf-8') as f:
            for question, question_dense_scores, question_dense_ids in zip(questions, dense_scores, dense_ids):
                valid = question_dense_ids >= 0
                question_dense_ids = question_dense_ids[valid]
                question_dense_scores = question_dense_scores[valid]

                sparse_scores = sparse_encoding(question["question"], vectorizer, X)
                sparse_ids, sparse_scores = top_k(sparse_scores,candidate_k)
                if method == "hybrid":
                    ranked = hybrid_rank(sparse_ids, sparse_scores, question_dense_ids, question_dense_scores, config["hybrid_alpha"])
                elif method == "rrf":
                    ranked = rrf_rank(sparse_ids, question_dense_ids, config["rrf_k"])
                seen_titles = set()
                titles = []
                for i in ranked:
                    title = chunk_set[int(i)]['title']
                    if title not in seen_titles:
                        seen_titles.add(title)
                        titles.append(title)
                row = {
                    "id": question['id'],
                    "titles": titles,
                }
                f.write(json.dumps(row)+"\n")



    # raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one retriever and write a run file.")
    parser.add_argument("--method", required=True, choices=("tfidf", "dpr", "hybrid", "rrf"))
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.method, args.questions, args.articles, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
