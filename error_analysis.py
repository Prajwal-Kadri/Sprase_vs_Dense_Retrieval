"""Print factual report data from existing retrieval runs; no retrieval is run.

Usage: python error_analysis.py pkadri
"""

import argparse
import json
from pathlib import Path

from assign_cases import assign_cases, load_rows


def selected_rankings(path, ids):
    rankings = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in ids:
                if row["id"] in rankings:
                    raise ValueError(f"Duplicate question {row['id']} in {path}")
                rankings[row["id"]] = row["titles"]
    missing = set(ids) - rankings.keys()
    if missing:
        raise ValueError(f"Missing questions in {path}: {sorted(missing)}")
    return rankings


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("andrew_id")
    args = parser.parse_args()

    questions = load_rows(root / "data/question_bank.jsonl")
    ids, multihop_ids = assign_cases(args.andrew_id, questions)
    by_id = {row["id"]: row for row in questions}
    runs = {
        method: selected_rankings(root / "runs" / filename, ids + multihop_ids)
        for method, filename in (
            ("TF-IDF", "tfidf.jsonl"),
            ("DPR", "dpr.jsonl"),
            ("Hybrid", "hybrid_0.5.jsonl"),
            ("RRF", "rrf.jsonl"),
        )
    }
    gaps = []
    missing_gold = []
    for query_id in ids:
        row = by_id[query_id]
        gold = set(row["title"])
        print(f"\n{'=' * 70}\n{query_id} ({row['type']})")
        print(row["question"])
        print("Gold titles:", json.dumps(row["title"], ensure_ascii=False))
        ranks = {}
        for method, rankings in runs.items():
            if method == "RRF":
                continue  # Error analysis requires TF-IDF, DPR, and hybrid only.
            titles = rankings[query_id]
            rank = next((i for i, title in enumerate(titles, 1) if title in gold), None)
            ranks[method] = rank
            print(f"\n{method} — top 3:")
            for i, title in enumerate(titles[:3], 1):
                print(f"  {i}. {title}")
            print("  First-gold rank:", rank if rank is not None else "not found")

        if ranks["TF-IDF"] is not None and ranks["DPR"] is not None:
            gap = abs(ranks["TF-IDF"] - ranks["DPR"])
            gaps.append((gap, query_id))
            print(f"\nTF-IDF/DPR rank gap: {gap}")
        else:
            missing_gold.append(query_id)
            print("\nTF-IDF/DPR rank gap: not numeric (gold not found in at least one ranking)")

    print("\nGap summary:")
    qualifying = [(gap, query_id) for gap, query_id in gaps if gap >= 5]
    if qualifying:
        for gap, query_id in sorted(qualifying, reverse=True):
            print(f"  {query_id}: gap {gap} (at least five positions)")
    elif gaps:
        gap, query_id = max(gaps)
        print(f"  No numeric gap reaches five. Largest: {query_id}, gap {gap}.")
    else:
        print("  No numeric gaps available.")
    if missing_gold:
        print("  Cases with a missing gold result:", ", ".join(missing_gold))
    print("Explain the observed differences yourself using query terms and article evidence.")

    print("\nMULTI-HOP TOP-10 COVERAGE")
    for query_id in multihop_ids:
        row = by_id[query_id]
        gold = list(dict.fromkeys(row["title"]))
        print(f"\n{'=' * 70}\n{query_id}")
        print(row["question"])
        print("Gold titles:", json.dumps(gold, ensure_ascii=False))
        for method, rankings in runs.items():
            titles = rankings[query_id]
            print(f"\n{method}:")
            for title in gold:
                rank = next((i for i, item in enumerate(titles, 1) if item == title), None)
                print(f"  {title}: rank {rank if rank is not None else 'not found'}")
            missing = [title for title in gold if title not in titles[:10]]
            print("  All gold articles in top 10:", "No" if missing else "Yes")
            print("  Missing from top 10:", json.dumps(missing, ensure_ascii=False) if missing else "none")
    print("\nRanks are 1-based positions in the complete submitted ranking.")
    print("'not found' means absent from that ranking; a rank above 10 is still missing from the top 10.")


if __name__ == "__main__":
    main()
