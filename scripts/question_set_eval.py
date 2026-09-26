"""Evaluate extracted question sets against a reference question set.

This evaluator sits above box-level crop metrics. It compares the final product
question set that a user sees, then reports missed questions, extras,
duplicates, merge suspects, and source trace coverage.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLACEHOLDER_MARKERS = (
    "未识别",
    "被遮挡",
    "仅见",
    "仅可见",
    "图示",
    "插图",
    "文字",
)


@dataclass
class Question:
    set_name: str
    ordinal: int
    row: dict[str, Any]
    stem: str
    number: str
    qtype: str
    text: str
    fingerprint: str
    simhash: str

    @property
    def question_id(self) -> str:
        return f"{self.set_name}:{self.ordinal}"

    @property
    def has_source(self) -> bool:
        return bool(
            self.row.get("src_filename")
            or self.row.get("source_filename")
            or self.row.get("source_image_id")
            or self.row.get("source_crop_id")
            or self.row.get("crop_filename")
        )

    @property
    def source_type(self) -> str:
        if self.row.get("source_type"):
            return str(self.row.get("source_type"))
        if self.row.get("source_crop_id") or self.row.get("crop_filename"):
            return "question_crop"
        if self.row.get("src_filename") or self.row.get("source_filename"):
            return "image"
        return "missing"


@dataclass
class QuestionList:
    path: Path
    source: dict[str, Any]
    questions: list[Question]


def read_payloads(path: Path) -> list[Any]:
    if not path.is_file():
        raise SystemExit(f"input file not found: {path}")
    text = read_text_auto(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        return rows
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON file {path}: {exc}") from exc
    return payload if isinstance(payload, list) else [payload]


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def display_text(question: dict[str, Any]) -> str:
    parts = [str(question.get("stem") or question.get("question") or question.get("content") or "")]
    figure_note = str(question.get("figure_note") or "")
    if figure_note:
        parts.append(figure_note)
    options = question.get("options")
    if isinstance(options, list):
        parts.extend(str(item) for item in options if item)
    return " ".join(parts).strip()


def is_question_like(value: Any) -> bool:
    return isinstance(value, dict) and bool(
        value.get("stem")
        or value.get("question")
        or value.get("content")
        or value.get("fingerprint")
        or value.get("simhash")
    )


def question_list_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, payload in enumerate(read_payloads(path)):
        if isinstance(payload, dict):
            questions = payload.get("questions") or payload.get("question_set") or payload.get("items")
            if isinstance(questions, list) and all(is_question_like(item) for item in questions):
                rows.append(
                    {
                        "path": str(path),
                        "row_index": row_index,
                        "type": payload.get("type") or "",
                        "count": payload.get("count") or payload.get("final_count") or len(questions),
                        "question_count": len(questions),
                        "questions": questions,
                        "row_keys": sorted(str(key) for key in payload.keys() if key != "questions"),
                    }
                )
        elif isinstance(payload, list) and all(is_question_like(item) for item in payload):
            rows.append(
                {
                    "path": str(path),
                    "row_index": row_index,
                    "type": "list",
                    "count": len(payload),
                    "question_count": len(payload),
                    "questions": payload,
                    "row_keys": [],
                }
            )
    return rows


def row_source_coverage(questions: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in questions
        if item.get("src_filename")
        or item.get("source_filename")
        or item.get("source_image_id")
        or item.get("source_crop_id")
        or item.get("crop_filename")
    )


def choose_row(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    if not rows:
        raise SystemExit("no question list rows found after filtering")
    normalized = selector.strip().lower()
    if normalized == "first":
        return rows[0]
    if normalized == "last":
        return rows[-1]
    if normalized == "best":
        return max(
            rows,
            key=lambda row: (
                int(row.get("question_count") or 0),
                row_source_coverage(row.get("questions") or []),
                sum(len(normalize_text(display_text(item))) for item in row.get("questions") or []),
                int(row.get("row_index") or 0),
            ),
        )
    if re.fullmatch(r"\d+", normalized):
        index = int(normalized)
        if not 0 <= index < len(rows):
            raise SystemExit(f"row selector {selector} is out of range for {len(rows)} question rows")
        return rows[index]
    raise SystemExit(f"unknown row selector: {selector}")


def load_question_list(path: Path, *, set_name: str, selector: str, row_type: str = "") -> QuestionList:
    rows = question_list_rows(path)
    if row_type:
        rows = [row for row in rows if str(row.get("type") or "") == row_type]
    selected = choose_row(rows, selector)
    questions: list[Question] = []
    for index, row in enumerate(selected["questions"], start=1):
        if not isinstance(row, dict):
            continue
        stem = display_text(row)
        questions.append(
            Question(
                set_name=set_name,
                ordinal=index,
                row=row,
                stem=stem,
                number=str(row.get("number") or "").strip(),
                qtype=str(row.get("qtype") or row.get("type") or "").strip(),
                text=normalize_text(stem),
                fingerprint=str(row.get("fingerprint") or "").strip(),
                simhash=str(row.get("simhash") or "").strip().lower(),
            )
        )
    source = {key: value for key, value in selected.items() if key != "questions"}
    return QuestionList(path=path, source=source, questions=questions)


def ngrams(text: str) -> set[str]:
    if not text:
        return set()
    n = 2 if len(text) < 16 else 3
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def simhash_hamming(a: str, b: str) -> int | None:
    if not re.fullmatch(r"[0-9a-fA-F]{16}", a or ""):
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{16}", b or ""):
        return None
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def placeholder_score(question: Question) -> int:
    return sum(1 for marker in PLACEHOLDER_MARKERS if marker in question.stem)


def common_prefix_len(a: str, b: str) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def pair_score(reference: Question, candidate: Question) -> dict[str, Any]:
    exact_fingerprint = bool(reference.fingerprint and reference.fingerprint == candidate.fingerprint)
    if exact_fingerprint:
        return {"score": 1.0, "text_similarity": 1.0, "reason": "fingerprint", "simhash_hamming": None}

    ref_text = reference.text
    cand_text = candidate.text
    sequence = difflib.SequenceMatcher(None, ref_text, cand_text).ratio() if ref_text or cand_text else 0.0
    grams = jaccard(ngrams(ref_text), ngrams(cand_text))
    score = max(sequence, grams)
    reason = "text"

    shorter = min(len(ref_text), len(cand_text))
    longer = max(len(ref_text), len(cand_text))
    containment = False
    if shorter >= 6 and ref_text and cand_text and (ref_text in cand_text or cand_text in ref_text):
        containment = True
        contain_score = min(0.97, max(0.78, shorter / max(1, longer) + 0.22))
        if shorter >= 10:
            contain_score = max(contain_score, 0.9)
        score = max(score, contain_score)
        reason = "containment"

    hamming = simhash_hamming(reference.simhash, candidate.simhash)
    if hamming is not None:
        if hamming <= 3:
            score = max(score, 0.93)
            reason = "simhash"
        elif hamming <= 8:
            score = max(score, 0.78)
            reason = "simhash_near"

    same_number = bool(reference.number and candidate.number and reference.number == candidate.number)
    number_mismatch = bool(reference.number and candidate.number and reference.number != candidate.number)
    same_qtype = bool(reference.qtype and candidate.qtype and reference.qtype == candidate.qtype)
    if same_number:
        score += 0.08
        if shorter >= 4 and containment:
            score = max(score, 0.9)
    elif number_mismatch:
        score -= 0.06
    elif reference.number or candidate.number:
        score += 0.015
    if same_qtype:
        score += 0.02

    prefix = common_prefix_len(ref_text, cand_text)
    if same_number and shorter >= 8 and prefix >= min(10, shorter):
        score = max(score, 0.72)

    if shorter < 4 and not same_number:
        score *= 0.5
    if placeholder_score(reference) and placeholder_score(candidate) and same_number:
        score = max(score, 0.62)

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "text_similarity": round(max(sequence, grams), 4),
        "reason": reason,
        "simhash_hamming": hamming,
        "same_number": same_number,
        "number_mismatch": number_mismatch,
        "same_qtype": same_qtype,
    }


def compact_question(question: Question) -> dict[str, Any]:
    return {
        "id": question.question_id,
        "ordinal": question.ordinal,
        "number": question.number,
        "qtype": question.qtype,
        "stem": question.stem,
        "fingerprint": question.fingerprint,
        "simhash": question.simhash,
        "src_filename": question.row.get("src_filename") or question.row.get("source_filename") or "",
        "source_type": question.source_type,
        "source_crop_id": question.row.get("source_crop_id") or "",
        "crop_filename": question.row.get("crop_filename") or "",
        "source_fallback_reason": question.row.get("source_fallback_reason") or "",
    }


def all_pair_scores(reference: list[Question], candidate: list[Question]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in reference:
        for cand in candidate:
            details = pair_score(ref, cand)
            rows.append(
                {
                    "reference_id": ref.question_id,
                    "candidate_id": cand.question_id,
                    "reference_ordinal": ref.ordinal,
                    "candidate_ordinal": cand.ordinal,
                    **details,
                }
            )
    return rows


def match_one_to_one(reference: list[Question], candidate: list[Question], min_score: float) -> dict[str, Any]:
    pairs = sorted(
        all_pair_scores(reference, candidate),
        key=lambda row: (
            float(row["score"]),
            bool(row.get("same_number")),
            -abs(int(row["reference_ordinal"]) - int(row["candidate_ordinal"])),
        ),
        reverse=True,
    )
    ref_by_id = {item.question_id: item for item in reference}
    cand_by_id = {item.question_id: item for item in candidate}
    used_refs: set[str] = set()
    used_candidates: set[str] = set()
    matches: list[dict[str, Any]] = []
    for pair in pairs:
        if float(pair["score"]) < min_score:
            continue
        ref_id = str(pair["reference_id"])
        cand_id = str(pair["candidate_id"])
        if ref_id in used_refs or cand_id in used_candidates:
            continue
        used_refs.add(ref_id)
        used_candidates.add(cand_id)
        ref = ref_by_id[ref_id]
        cand = cand_by_id[cand_id]
        matches.append(
            {
                **pair,
                "reference": compact_question(ref),
                "candidate": compact_question(cand),
            }
        )
    return {"matches": matches, "used_refs": used_refs, "used_candidates": used_candidates, "pairs": pairs}


def best_pair(question: Question, others: list[Question], *, reverse: bool = False) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for other in others:
        details = pair_score(other, question) if reverse else pair_score(question, other)
        row = {
            "reference_id": other.question_id if reverse else question.question_id,
            "candidate_id": question.question_id if reverse else other.question_id,
            "score": details["score"],
            "reason": details["reason"],
            "text_similarity": details["text_similarity"],
        }
        if best is None or float(row["score"]) > float(best["score"]):
            best = row
    return best


def coverage_report(reference: list[Question], candidate: list[Question], min_score: float) -> dict[str, Any]:
    coverage_rows: list[dict[str, Any]] = []
    covered_ref_ids: set[str] = set()
    candidate_to_refs: dict[str, list[dict[str, Any]]] = {}
    for ref in reference:
        scored = []
        for cand in candidate:
            details = pair_score(ref, cand)
            if float(details["score"]) >= min_score:
                scored.append({"candidate": compact_question(cand), **details})
        scored.sort(key=lambda row: float(row["score"]), reverse=True)
        if scored:
            covered_ref_ids.add(ref.question_id)
            for row in scored:
                candidate_to_refs.setdefault(row["candidate"]["id"], []).append(
                    {
                        "reference": compact_question(ref),
                        "score": row["score"],
                        "reason": row["reason"],
                    }
                )
        coverage_rows.append({"reference": compact_question(ref), "covered_by": scored[:5]})

    merge_candidates = []
    for candidate_id, refs in candidate_to_refs.items():
        if len(refs) <= 1:
            continue
        cand = next(item for item in candidate if item.question_id == candidate_id)
        refs.sort(key=lambda row: float(row["score"]), reverse=True)
        merge_candidates.append({"candidate": compact_question(cand), "covered_references": refs})
    merge_candidates.sort(key=lambda row: len(row["covered_references"]), reverse=True)
    return {
        "coverage_rows": coverage_rows,
        "covered_ref_ids": covered_ref_ids,
        "merge_candidates": merge_candidates,
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def duplicate_clusters(questions: list[Question], min_score: float) -> list[dict[str, Any]]:
    if len(questions) < 2:
        return []
    groups = UnionFind(len(questions))
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for left in range(len(questions)):
        for right in range(left + 1, len(questions)):
            details = pair_score(questions[left], questions[right])
            if float(details["score"]) >= min_score:
                groups.union(left, right)
                edges[(left, right)] = details
    buckets: dict[int, list[int]] = {}
    for index in range(len(questions)):
        buckets.setdefault(groups.find(index), []).append(index)
    clusters: list[dict[str, Any]] = []
    for indexes in buckets.values():
        if len(indexes) <= 1:
            continue
        pair_edges = []
        for i, left in enumerate(indexes):
            for right in indexes[i + 1 :]:
                details = edges.get((min(left, right), max(left, right)))
                if details:
                    pair_edges.append(
                        {
                            "left": questions[left].question_id,
                            "right": questions[right].question_id,
                            **details,
                        }
                    )
        clusters.append(
            {
                "questions": [compact_question(questions[index]) for index in indexes],
                "pair_edges": pair_edges,
            }
        )
    clusters.sort(key=lambda row: len(row["questions"]), reverse=True)
    return clusters


def source_summary(questions: list[Question]) -> dict[str, Any]:
    type_counts = Counter(question.source_type for question in questions)
    fallback_counts = Counter(
        str(question.row.get("source_fallback_reason") or "")
        for question in questions
        if question.row.get("source_fallback_reason")
    )
    crop_safety_reasons: Counter[str] = Counter()
    weak_key_count = 0
    low_confidence_count = 0
    expanded_count = 0
    input_index_count = 0
    for question in questions:
        key = str(
            question.row.get("client_question_key")
            or question.row.get("question_key")
            or ""
        )
        if key.startswith("layout:"):
            weak_key_count += 1
        if question.row.get("source_input_index") or question.row.get("input_index"):
            input_index_count += 1
        confidence = question.row.get("confidence")
        try:
            if confidence is not None and float(confidence) < 0.45:
                low_confidence_count += 1
        except (TypeError, ValueError):
            pass
        safety = question.row.get("crop_safety")
        if isinstance(safety, str):
            try:
                safety = json.loads(safety)
            except json.JSONDecodeError:
                safety = {}
        if isinstance(safety, dict):
            reason_values = safety.get("reasons") or safety.get("risk_reasons") or []
            if isinstance(reason_values, str):
                reason_values = [reason_values]
            for reason in reason_values:
                crop_safety_reasons[str(reason)] += 1
            if safety.get("did_expand") or safety.get("server_rect_expanded") or safety.get("expanded"):
                expanded_count += 1
    missing = [compact_question(question) for question in questions if not question.has_source]
    source_files = {
        str(question.row.get("src_filename") or question.row.get("source_filename") or "")
        for question in questions
        if question.row.get("src_filename") or question.row.get("source_filename")
    }
    return {
        "with_source": len(questions) - len(missing),
        "without_source": len(missing),
        "source_coverage": round((len(questions) - len(missing)) / max(1, len(questions)), 4),
        "source_type_counts": dict(sorted(type_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        "crop_safety_reason_counts": dict(sorted(crop_safety_reasons.items())),
        "weak_layout_key_count": weak_key_count,
        "low_confidence_count": low_confidence_count,
        "expanded_crop_count": expanded_count,
        "source_input_index_count": input_index_count,
        "source_input_index_missing": len(questions) - input_index_count,
        "unique_source_file_count": len(source_files),
        "missing_source_questions": missing,
    }


def number_summary(questions: list[Question]) -> dict[str, Any]:
    numbers = [question.number for question in questions]
    present = [number for number in numbers if number]
    counts = Counter(present)
    numeric_values: list[int] = []
    for number in present:
        match = re.search(r"\d+", number)
        if match:
            numeric_values.append(int(match.group(0)))
    duplicate_numbers = {number: count for number, count in counts.items() if count > 1}
    monotonic_breaks = 0
    for left, right in zip(numeric_values, numeric_values[1:]):
        if right < left:
            monotonic_breaks += 1
    jumps = 0
    unique_numeric = sorted(set(numeric_values))
    for left, right in zip(unique_numeric, unique_numeric[1:]):
        if right - left > 1:
            jumps += 1
    return {
        "missing_number_count": len(questions) - len(present),
        "duplicate_number_counts": duplicate_numbers,
        "duplicate_number_question_count": sum(count - 1 for count in duplicate_numbers.values()),
        "numeric_monotonic_breaks": monotonic_breaks,
        "numeric_gap_count": jumps,
    }


def content_summary(questions: list[Question]) -> dict[str, Any]:
    short_count = sum(1 for question in questions if 0 < len(question.text) < 8)
    empty_count = sum(1 for question in questions if not question.text)
    placeholder_count = sum(1 for question in questions if placeholder_score(question))
    qtype_counts = Counter(question.qtype or "missing" for question in questions)
    generic_type_count = qtype_counts.get("其它", 0) + qtype_counts.get("其他", 0)
    with_options = sum(1 for question in questions if isinstance(question.row.get("options"), list) and question.row.get("options"))
    with_figure_note = sum(1 for question in questions if question.row.get("figure_note"))
    with_student_answer = sum(1 for question in questions if question.row.get("has_student_answer") or question.row.get("student_answer"))
    return {
        "empty_text_count": empty_count,
        "short_text_count": short_count,
        "placeholder_count": placeholder_count,
        "generic_qtype_count": generic_type_count,
        "qtype_counts": dict(sorted(qtype_counts.items())),
        "with_options_count": with_options,
        "with_figure_note_count": with_figure_note,
        "with_student_answer_count": with_student_answer,
    }


def stem_hash(question: Question) -> str:
    return hashlib.sha1(question.text.encode("utf-8")).hexdigest() if question.text else ""


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_question_list(
        args.reference,
        set_name="reference",
        selector=args.reference_row,
        row_type=args.reference_type,
    )
    candidate = load_question_list(
        args.candidate,
        set_name="candidate",
        selector=args.candidate_row,
        row_type=args.candidate_type,
    )
    match_result = match_one_to_one(reference.questions, candidate.questions, args.min_score)
    coverage = coverage_report(reference.questions, candidate.questions, args.coverage_min_score)
    ref_duplicates = duplicate_clusters(reference.questions, args.duplicate_min_score)
    cand_duplicates = duplicate_clusters(candidate.questions, args.duplicate_min_score)

    matched_count = len(match_result["matches"])
    ref_count = len(reference.questions)
    cand_count = len(candidate.questions)
    precision = matched_count / max(1, cand_count)
    recall = matched_count / max(1, ref_count)
    f1 = (2 * precision * recall / max(0.000001, precision + recall)) if precision + recall else 0.0
    coverage_recall = len(coverage["covered_ref_ids"]) / max(1, ref_count)

    missed_rows = []
    for ref in reference.questions:
        if ref.question_id in match_result["used_refs"]:
            continue
        best = best_pair(ref, candidate.questions)
        covered = ref.question_id in coverage["covered_ref_ids"]
        missed_rows.append(
            {
                "reference": compact_question(ref),
                "covered_by_any_candidate": covered,
                "best_candidate_pair": best,
                "stem_hash": stem_hash(ref),
            }
        )

    extra_rows = []
    for cand in candidate.questions:
        if cand.question_id in match_result["used_candidates"]:
            continue
        extra_rows.append(
            {
                "candidate": compact_question(cand),
                "best_reference_pair": best_pair(cand, reference.questions, reverse=True),
                "stem_hash": stem_hash(cand),
            }
        )

    number_mismatch_matches = [
        match for match in match_result["matches"] if match.get("number_mismatch")
    ]
    summary = {
        "reference": reference.source,
        "candidate": candidate.source,
        "thresholds": {
            "min_score": args.min_score,
            "coverage_min_score": args.coverage_min_score,
            "duplicate_min_score": args.duplicate_min_score,
        },
        "counts": {
            "reference_count": ref_count,
            "candidate_count": cand_count,
            "count_delta": cand_count - ref_count,
            "matched": matched_count,
            "missed": len(missed_rows),
            "extra": len(extra_rows),
            "covered_reference_count": len(coverage["covered_ref_ids"]),
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "coverage_recall": round(coverage_recall, 4),
        },
        "duplicates": {
            "reference_duplicate_clusters": len(ref_duplicates),
            "reference_duplicate_question_count": sum(len(row["questions"]) - 1 for row in ref_duplicates),
            "candidate_duplicate_clusters": len(cand_duplicates),
            "candidate_duplicate_question_count": sum(len(row["questions"]) - 1 for row in cand_duplicates),
        },
        "merge_suspects": {
            "candidate_count": len(coverage["merge_candidates"]),
            "covered_reference_overlaps": sum(len(row["covered_references"]) - 1 for row in coverage["merge_candidates"]),
        },
        "number_mismatch_count": len(number_mismatch_matches),
        "placeholder_counts": {
            "reference": sum(1 for question in reference.questions if placeholder_score(question)),
            "candidate": sum(1 for question in candidate.questions if placeholder_score(question)),
        },
        "numbers": {
            "reference": number_summary(reference.questions),
            "candidate": number_summary(candidate.questions),
        },
        "content": {
            "reference": content_summary(reference.questions),
            "candidate": content_summary(candidate.questions),
        },
        "source": {
            "reference": source_summary(reference.questions),
            "candidate": source_summary(candidate.questions),
        },
    }
    return {
        "summary": summary,
        "matches": match_result["matches"],
        "missed": missed_rows,
        "extra": extra_rows,
        "coverage": coverage["coverage_rows"],
        "merge_suspects": coverage["merge_candidates"],
        "reference_duplicates": ref_duplicates,
        "candidate_duplicates": cand_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate product-level question-set extraction.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference JSON/JSONL containing a questions list.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate JSON/JSONL containing a questions list.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-set-eval"))
    parser.add_argument("--reference-type", default="", help="Optional row type filter for the reference JSONL, e.g. product_qset.")
    parser.add_argument("--candidate-type", default="", help="Optional row type filter for the candidate JSONL.")
    parser.add_argument("--reference-row", default="best", help="Question-list row selector: best, first, last, or zero-based index.")
    parser.add_argument("--candidate-row", default="best", help="Question-list row selector: best, first, last, or zero-based index.")
    parser.add_argument("--min-score", type=float, default=0.58, help="Minimum pair score for one-to-one matching.")
    parser.add_argument("--coverage-min-score", type=float, default=0.55, help="Minimum score for many-to-one coverage recall.")
    parser.add_argument("--duplicate-min-score", type=float, default=0.82, help="Minimum score for within-set duplicate clusters.")
    parser.add_argument("--clean", action="store_true", help="Remove old output files before writing.")
    args = parser.parse_args()

    if args.clean and args.out.exists():
        for child in args.out.glob("*"):
            if child.is_file():
                child.unlink()
    result = evaluate(args)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", result["summary"])
    write_jsonl(args.out / "matches.jsonl", result["matches"])
    write_jsonl(args.out / "missed.jsonl", result["missed"])
    write_jsonl(args.out / "extra.jsonl", result["extra"])
    write_jsonl(args.out / "coverage.jsonl", result["coverage"])
    write_jsonl(args.out / "merge_suspects.jsonl", result["merge_suspects"])
    write_jsonl(args.out / "reference_duplicates.jsonl", result["reference_duplicates"])
    write_jsonl(args.out / "candidate_duplicates.jsonl", result["candidate_duplicates"])
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
