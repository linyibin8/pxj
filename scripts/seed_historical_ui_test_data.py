#!/usr/bin/env python3
"""Seed pxj UI-test data from historical xue photos.

The script intentionally leaves created sessions, mistakes, and uploaded images
in place so simulator runs can be inspected after completion.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover - runtime host check
    raise SystemExit("python3 requests package is required on the seed host") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://pxj.evowit.com")
    parser.add_argument("--source-dir", default="/home/ydz/services/xue/backend/data/images")
    parser.add_argument("--marker", default="PXJ-HISTORY-SIM-20260629")
    parser.add_argument("--email", default="pxj-sim-history-20260629@example.com")
    parser.add_argument("--password", default=os.environ.get("PXJ_SEED_PASSWORD", ""))
    parser.add_argument("--image-count", type=int, default=5)
    return parser.parse_args()


def checked_json(response: requests.Response, context: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text[:500]}
    if response.status_code >= 400:
        raise RuntimeError(f"{context} failed: HTTP {response.status_code}: {payload}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} returned non-object JSON: {payload!r}")
    return payload


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def register_or_login(base_url: str, email: str, password: str) -> dict[str, Any]:
    register_payload = {
        "email": email,
        "password": password,
        "display_name": "PXJ Simulator UI",
        "account_name": "PXJ Simulator Historical Photo Regression",
        "student_name": "Historical Photo Student",
    }
    response = requests.post(f"{base_url}/api/auth/register", json=register_payload, timeout=30)
    if response.status_code == 409:
        response = requests.post(
            f"{base_url}/api/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
        payload = checked_json(response, "login")
        payload["auth_mode"] = "login"
        return payload
    payload = checked_json(response, "register")
    payload["auth_mode"] = "register"
    return payload


def choose_images(source_dir: Path, count: int) -> list[Path]:
    images = sorted(
        [path for path in source_dir.glob("*.jpg") if path.is_file() and path.stat().st_size > 0],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if len(images) < count:
        raise RuntimeError(f"only found {len(images)} jpg files under {source_dir}, need {count}")
    return images[:count]


def create_session(base_url: str, token: str, marker: str) -> str:
    response = requests.post(
        f"{base_url}/api/sessions",
        headers=auth_headers(token),
        data={
            "device_id": "ios-simulator",
            "mode": "burst",
            "title": f"{marker} historical xue photo session",
            "student_goal": f"{marker} use original xue historical photos for simulator UI regression",
            "report_style": "UI regression seed; retain data after test run.",
            "assistant_focus": "History list and mistake-book rendering with real uploaded photos.",
        },
        timeout=30,
    )
    payload = checked_json(response, "create session")
    session_id = str(payload.get("session_id") or "")
    if not session_id:
        raise RuntimeError(f"create session returned no session_id: {payload}")
    return session_id


def upload_images(base_url: str, token: str, session_id: str, marker: str, image_paths: list[Path]) -> dict[str, Any]:
    capture_meta = [
        {
            "sequence_index": index,
            "page_hint": f"{marker} historical-photo-page-{index + 1}",
            "question_hint": f"{marker} historical-photo-question-{index + 1}",
            "source": "xue historical photo archive",
            "source_path": str(path),
        }
        for index, path in enumerate(image_paths)
    ]
    with contextlib.ExitStack() as stack:
        files = [
            ("images", (path.name, stack.enter_context(path.open("rb")), "image/jpeg"))
            for path in image_paths
        ]
        response = requests.post(
            f"{base_url}/api/sessions/{session_id}/batches",
            headers=auth_headers(token),
            data={
                "device_id": "ios-simulator",
                "environment": "mac simulator historical-photo regression",
                "capture_meta": json.dumps(capture_meta, ensure_ascii=False),
            },
            files=files,
            timeout=120,
        )
    return checked_json(response, "upload images")


def create_mistake(base_url: str, token: str, session_id: str, marker: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/api/sessions/{session_id}/mistakes",
        headers=auth_headers(token),
        json={
            "title": f"{marker} seeded historical-photo mistake",
            "question_text": f"{marker}: seeded from original xue historical photos for iPad mistake-detail UI testing.",
            "student_answer": "Seeded simulator answer: skipped one reasoning step.",
            "expected_answer": "Seeded simulator reference: show the missing reasoning step and final answer.",
            "error_reason": "Historical-photo simulator regression data; verifies non-empty mistake detail.",
            "knowledge_points": ["historical-photo-regression", "simulator-ui", marker],
            "subject": "Math",
            "page_ref": "xue/backend/data/images",
            "question_ref": f"{marker}-manual-mistake",
            "location_ref": "seed_historical_ui_test_data.py",
            "error_type": "process",
            "correction": "Retain this item so future simulator runs can inspect the detail pane.",
            "next_action": "Open iPad Mistakes and verify review actions are visible.",
        },
        timeout=30,
    )
    return checked_json(response, "create mistake")


def verify(base_url: str, token: str, marker: str) -> dict[str, Any]:
    sessions = checked_json(
        requests.get(f"{base_url}/api/sessions", headers=auth_headers(token), timeout=30),
        "list sessions",
    )
    review = checked_json(
        requests.get(
            f"{base_url}/api/review-queue",
            headers=auth_headers(token),
            params={"due_only": "false", "q": marker, "page_size": "20"},
            timeout=30,
        ),
        "review queue",
    )
    session_items = sessions.get("sessions") or sessions.get("items") or []
    review_items = review.get("items") or review.get("mistakes") or []
    return {
        "matching_sessions": sum(1 for item in session_items if marker in json.dumps(item, ensure_ascii=False)),
        "matching_review_items": sum(1 for item in review_items if marker in json.dumps(item, ensure_ascii=False)),
    }


def main() -> None:
    args = parse_args()
    if not args.password:
        raise SystemExit("PXJ_SEED_PASSWORD or --password is required")
    base_url = args.base_url.rstrip("/")
    source_dir = Path(args.source_dir)
    selected = choose_images(source_dir, args.image_count)
    auth = register_or_login(base_url, args.email, args.password)
    token = str(auth.get("access_token") or "")
    if not token:
        raise RuntimeError("authentication returned no access token")
    session_id = create_session(base_url, token, args.marker)
    batch = upload_images(base_url, token, session_id, args.marker, selected)
    mistake = create_mistake(base_url, token, session_id, args.marker)
    verification = verify(base_url, token, args.marker)
    print(
        json.dumps(
            {
                "marker": args.marker,
                "email": args.email,
                "auth_mode": auth.get("auth_mode"),
                "session_id": session_id,
                "uploaded_image_count": batch.get("image_count"),
                "analysis_image_count": batch.get("analysis_image_count"),
                "mistake_id": (mistake.get("mistake") or {}).get("id"),
                "selected_images": [str(path) for path in selected],
                "verification": verification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
