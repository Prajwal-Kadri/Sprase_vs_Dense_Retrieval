"""Official HW2 scorer. Report numbers are checked against this file. 
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAIN_TYPES = ("factoid", "complex", "multi_hop")
KS = (1, 5, 10)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(row)
    return rows


def _distinct(titles) -> list[str]:
    if not isinstance(titles, list):
        return []
    seen = []
    for title in titles:
        if isinstance(title, str) and title not in seen:
            seen.append(title)
    return seen


def recall_at(gold: list[str], ranked: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(gold) & set(ranked[:k])) / len(set(gold))


def all_recall_at(gold: list[str], ranked: list[str], k: int) -> float:
    return 1.0 if set(gold) <= set(ranked[:k]) else 0.0


def mrr_at(gold: list[str], ranked: list[str], k: int = 10) -> float:
    gold_set = set(gold)
    for rank, title in enumerate(ranked[:k], start=1):
        if title in gold_set:
            return 1.0 / rank
    return 0.0


def near_miss_at(gold: list[str], ranked: list[str], k: int) -> float:
    return 1.0 if set(gold) & set(ranked[:k]) else 0.0


def validate_run(gold_rows: list[dict], run_rows: list[dict]) -> list[str]:
    errors = []
    gold_ids = []
    for row in gold_rows:
        gold_ids.append(row.get("id"))
    gold_id_set = set(gold_ids)
    if len(gold_id_set) != len(gold_ids):
        errors.append("gold file has duplicate ids")
    seen = set()
    for index, row in enumerate(run_rows, start=1):
        query_id = row.get("id")
        if not isinstance(query_id, str) or not query_id:
            errors.append(f"run row {index} is missing id")
            continue
        if query_id in seen:
            errors.append(f"duplicate query id {query_id}")
        seen.add(query_id)
        if query_id not in gold_id_set:
            errors.append(f"unknown query id {query_id}")
        titles = row.get("titles")
        if not isinstance(titles, list) or not all(isinstance(title, str) for title in titles):
            errors.append(f"run row {query_id} titles must be a list of strings")
            continue
        if len(titles) != len(set(titles)):
            errors.append(f"duplicate article in run for {query_id}")
    missing = [query_id for query_id in gold_ids if query_id not in seen]
    if missing:
        errors.append(f"missing {len(missing)} query ids, first {missing[0]}")
    return errors


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _pack(recalls: dict[int, list[float]], mrrs: list[float]) -> dict:
    packed = {f"recall@{k}": _mean(recalls[k]) for k in KS}
    packed["mrr@10"] = _mean(mrrs)
    packed["n"] = len(mrrs)
    return packed


def evaluate(gold_rows: list[dict], run_rows: list[dict]) -> dict:
    rankings = {}
    for row in run_rows:
        query_id = row.get("id")
        titles = row.get("titles")
        if isinstance(query_id, str) and isinstance(titles, list):
            rankings.setdefault(query_id, _distinct(titles))

    overall_recall = {k: [] for k in KS}
    overall_mrr = []
    by_type_recall = {qtype: {k: [] for k in KS} for qtype in MAIN_TYPES}
    by_type_mrr = {qtype: [] for qtype in MAIN_TYPES}
    all_recall = {k: [] for k in KS}
    near = {k: [] for k in KS}
    warnings = []

    for row in gold_rows:
        query_id = row.get("id")
        qtype = row.get("type")
        gold = _distinct(row.get("title"))
        ranked = rankings.get(query_id, [])
        if query_id not in rankings:
            warnings.append(f"missing run row for {query_id}")
        if qtype == "unanswerable":
            for k in KS:
                near[k].append(near_miss_at(gold, ranked, k))
            continue
        if qtype not in MAIN_TYPES:
            warnings.append(f"skipping unknown type for {query_id}")
            continue
        for k in KS:
            value = recall_at(gold, ranked, k)
            overall_recall[k].append(value)
            by_type_recall[qtype][k].append(value)
            if qtype == "multi_hop":
                all_recall[k].append(all_recall_at(gold, ranked, k))
        mrr = mrr_at(gold, ranked, 10)
        overall_mrr.append(mrr)
        by_type_mrr[qtype].append(mrr)

    return {
        "overall": _pack(overall_recall, overall_mrr),
        "by_type": {qtype: _pack(by_type_recall[qtype], by_type_mrr[qtype]) for qtype in MAIN_TYPES},
        "all_recall": {f"all_recall@{k}": _mean(all_recall[k]) for k in KS},
        "near_miss": {f"near_miss@{k}": _mean(near[k]) for k in KS},
        "num_scored": len(overall_mrr),
        "num_unanswerable": len(near[1]),
        "warnings": warnings,
    }


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_report(metrics: dict) -> str:
    lines = []
    overall = metrics["overall"]
    lines.append(
        "overall  "
        + "  ".join(f"{name}={_fmt(overall[name])}" for name in ("recall@1", "recall@5", "recall@10", "mrr@10"))
        + f"  n={overall['n']}"
    )
    for qtype, block in metrics["by_type"].items():
        lines.append(
            f"{qtype:12} "
            + "  ".join(f"{name}={_fmt(block[name])}" for name in ("recall@1", "recall@5", "recall@10", "mrr@10"))
            + f"  n={block['n']}"
        )
    all_recall = metrics["all_recall"]
    lines.append(
        "all_recall  "
        + "  ".join(f"{name}={_fmt(all_recall[name])}" for name in ("all_recall@1", "all_recall@5", "all_recall@10"))
    )
    near = metrics["near_miss"]
    lines.append(
        "near_miss  "
        + "  ".join(f"{name}={_fmt(near[name])}" for name in ("near_miss@1", "near_miss@5", "near_miss@10"))
        + f"  n={metrics['num_unanswerable']}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a retrieval run against question_bank.jsonl.")
    parser.add_argument("gold", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    gold_rows = load_jsonl(args.gold)
    run_rows = load_jsonl(args.run)
    errors = validate_run(gold_rows, run_rows)
    if args.validate:
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("valid")
        return 0
    if errors:
        print("\n".join(errors), file=sys.stderr)
    metrics = evaluate(gold_rows, run_rows)
    print(format_report(metrics))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
