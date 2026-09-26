"""Sync a homework evidence ledger simulator output directory to the backend.

This is a diagnostic bridge for the independent homework-ledger feature:
run scripts/homework_evidence_ledger_sim.py locally, then upload its keyframes,
evidence crops, and manifest through /api/homework-ledger.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any

import httpx


def read_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def rel_from_out(path_text: str, out_dir: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return path.resolve().relative_to(out_dir.resolve()).as_posix()
    except Exception:
        return path.as_posix().replace("\\", "/")


def simulator_manifest(out_dir: Path) -> dict[str, Any]:
    frames = read_jsonl(out_dir / "frames.jsonl")
    for frame in frames:
        normalized = rel_from_out(str(frame.get("normalized_path") or ""), out_dir)
        if normalized:
            frame["filename"] = normalized
            frame["upload_ref"] = normalized
    evidence = read_json(out_dir / "question_evidence.json", [])
    for item in evidence:
        crop = rel_from_out(str(item.get("best_crop_filename") or ""), out_dir)
        if crop:
            item["best_crop_filename"] = crop
            item["upload_ref"] = crop
    cards = read_json(out_dir / "question_cards.json", [])
    for card in cards:
        for asset in card.get("figure_assets") or []:
            if not isinstance(asset, dict):
                continue
            filename = rel_from_out(str(asset.get("filename") or ""), out_dir)
            if filename:
                asset["filename"] = filename
    metrics = read_json(out_dir / "metrics.json", {})
    return {
        "source": "homework_evidence_ledger_sim",
        "out_dir": str(out_dir),
        "frames": frames,
        "page_episodes": read_json(out_dir / "page_episodes.json", []),
        "question_evidence": evidence,
        "question_cards": cards,
        "metrics": metrics,
    }


def file_tuple(path: Path) -> tuple[str, tuple[str, object, str]]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.as_posix(), (path.as_posix(), path.open("rb"), mime)


def sync(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    manifest = simulator_manifest(out_dir)
    run_id = args.run_id or f"ledger_{uuid.uuid4().hex[:12]}"
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    timeout = httpx.Timeout(args.timeout, connect=10)
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=timeout, headers=headers, trust_env=False) as client:
        create_response = client.post(
            "/api/homework-ledger/runs",
            json={"run_id": run_id, "title": args.title, "device_id": args.device_id},
        )
        create_response.raise_for_status()
        files: list[tuple[str, tuple[str, object, str]]] = []
        open_handles: list[object] = []
        try:
            for frame in manifest["frames"]:
                if not frame.get("accepted", True):
                    continue
                rel = str(frame.get("filename") or "")
                path = out_dir / rel
                if path.is_file():
                    name, payload = file_tuple(path)
                    files.append(("frames", payload))
                    open_handles.append(payload[1])
            for item in manifest["question_evidence"]:
                rel = str(item.get("best_crop_filename") or "")
                path = out_dir / rel
                if path.is_file():
                    name, payload = file_tuple(path)
                    files.append(("crops", payload))
                    open_handles.append(payload[1])
            sync_response = client.post(
                f"/api/homework-ledger/runs/{run_id}/sync",
                data={"manifest": json.dumps(manifest, ensure_ascii=False)},
                files=files,
            )
            sync_response.raise_for_status()
        finally:
            for handle in open_handles:
                try:
                    handle.close()
                except Exception:
                    pass
        finish_response = client.post(
            f"/api/homework-ledger/runs/{run_id}/finish",
            json={"metrics": manifest.get("metrics") or {}},
        )
        finish_response.raise_for_status()
        payload_response = client.get(f"/api/homework-ledger/runs/{run_id}")
        payload_response.raise_for_status()
    payload = payload_response.json()
    return {
        "run_id": run_id,
        "create": create_response.json(),
        "sync": sync_response.json(),
        "finish": finish_response.json(),
        "counts": {
            "frames": len(payload.get("frames") or []),
            "page_episodes": len(payload.get("page_episodes") or []),
            "evidence": len(payload.get("evidence") or []),
            "cards": len(payload.get("cards") or []),
        },
        "html_url": f"{args.base_url.rstrip('/')}/api/homework-ledger/runs/{run_id}/html",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Simulator output directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--title", default="作业证据账本模拟回放")
    parser.add_argument("--device-id", default="mac-simulator")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    result = sync(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
