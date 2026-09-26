import html
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse


HOMEWORK_LEDGER_ASSET_ROOT = "homework_ledger"
HOMEWORK_LEDGER_HTML_LIMIT = 240_000


def _json_object(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return dict(parsed)
    return {}


def _json_array(value: object) -> list:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return list(parsed)
    return []


def _meta_text(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_list(manifest: dict, *keys: str) -> list[dict]:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _parse_manifest(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "invalid manifest json") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "manifest must be an object")
    return data


def _file_map(uploads: list[UploadFile] | None) -> dict[str, UploadFile]:
    mapped: dict[str, UploadFile] = {}
    for index, upload in enumerate(uploads or []):
        original = str(upload.filename or "").replace("\\", "/").strip().lstrip("/")
        if original:
            mapped.setdefault(original, upload)
            mapped.setdefault(Path(original).name, upload)
        mapped.setdefault(str(index), upload)
    return mapped


def _upload_ref(item: dict, *keys: str, fallback: str = "") -> str:
    for key in keys:
        value = str(item.get(key) or "").replace("\\", "/").strip().lstrip("/")
        if value:
            return value
    return fallback


def _safe_asset_rel(rel_path: str, default_folder: str, fallback_name: str) -> str:
    clean = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
    parts = [part for part in clean.split("/") if part and part not in {".", ".."}]
    if not parts:
        parts = [fallback_name]
    if default_folder in parts:
        start = len(parts) - 1 - list(reversed(parts)).index(default_folder)
        parts = parts[start:]
    elif default_folder == "frames" and "normalized_frames" in parts:
        parts = ["frames", parts[-1]]
    elif len(parts) == 1 or parts[0] not in {"frames", "crops"}:
        parts = [default_folder, parts[-1]]
    return "/".join(parts[:3])


def _frame_id(item: dict, fallback_index: int) -> str:
    return _truncate(_meta_text(item, "id", "frame_id", "frameId") or f"frame_{fallback_index:04d}", 80)


def _evidence_id(item: dict, fallback_index: int) -> str:
    return _truncate(_meta_text(item, "id", "evidence_id", "evidenceId") or f"qev_{fallback_index:04d}", 80)


def _card_id(item: dict, fallback_index: int) -> str:
    return _truncate(_meta_text(item, "id", "card_id", "cardId") or f"q_{fallback_index:04d}", 80)


def _rect_payload(item: dict) -> dict:
    rect = _json_object(item.get("canonical_rect") or item.get("canonicalRect") or item.get("rect") or {})
    if rect and "width" not in rect and "w" in rect:
        rect["width"] = rect.get("w")
    if rect and "height" not in rect and "h" in rect:
        rect["height"] = rect.get("h")
    return rect


def register_homework_ledger_routes(
    app,
    *,
    init_db: Callable[[], None],
    principal_from_request: Callable[[Request], dict],
    connect: Callable,
    utc_now: Callable[[], str],
    get_settings: Callable,
    resolve_student_profile: Callable[[str, str], str],
    clean_user_text: Callable,
    json_dumps: Callable[[object], str],
    emit_log: Callable,
) -> None:
    def ledger_root() -> Path:
        path = get_settings().data_dir / HOMEWORK_LEDGER_ASSET_ROOT
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(run_id: str) -> Path:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id or ""))[:120]
        if not safe_run_id:
            raise HTTPException(400, "invalid run id")
        path = (ledger_root() / safe_run_id).resolve()
        root = ledger_root().resolve()
        if not path.is_relative_to(root):
            raise HTTPException(400, "invalid run id")
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(exist_ok=True)
        (path / "crops").mkdir(exist_ok=True)
        return path

    def asset_file_path(run_id: str, rel_path: str) -> Path:
        clean_rel = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
        if not clean_rel or ".." in Path(clean_rel).parts:
            raise HTTPException(400, "invalid asset path")
        base = run_dir(run_id).resolve()
        path = (base / clean_rel).resolve()
        if not path.is_relative_to(base):
            raise HTTPException(400, "invalid asset path")
        if not path.is_file():
            raise HTTPException(404, "asset not found")
        return path

    def asset_url(run_id: str, rel_path: str) -> str:
        clean_rel = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
        if not clean_rel:
            return ""
        return f"/api/homework-ledger/runs/{quote(run_id, safe='')}/assets/{quote(clean_rel, safe='/')}"

    async def save_upload(
        upload: UploadFile,
        *,
        run_id: str,
        rel_path: str,
        default_folder: str,
        fallback_prefix: str,
    ) -> tuple[str, int]:
        ext = Path(upload.filename or rel_path or "asset.jpg").suffix.lower() or ".jpg"
        fallback_name = f"{fallback_prefix}_{uuid.uuid4().hex[:10]}{ext}"
        safe_rel = _safe_asset_rel(rel_path, default_folder, fallback_name)
        target = run_dir(run_id) / safe_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            upload.file.seek(0)
        except Exception:
            pass
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        return safe_rel, target.stat().st_size

    def require_run(conn, run_id: str, principal: dict) -> dict:
        if principal.get("authenticated") or get_settings().auth_required:
            row = conn.execute(
                "SELECT * FROM homework_ledger_runs WHERE id=? AND account_id=?",
                (run_id, principal["account_id"]),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM homework_ledger_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(404, "homework ledger run not found")
        return dict(row)

    def card_row(row: dict, run_id: str) -> dict:
        item = dict(row)
        item["evidence_ids"] = _json_array(item.pop("evidence_ids", "[]"))
        item["figure_assets"] = _json_array(item.pop("figure_assets", "[]"))
        item["review_flags"] = _json_array(item.pop("review_flags", "[]"))
        item["payload"] = _json_object(item.get("payload"))
        for asset in item["figure_assets"]:
            if isinstance(asset, dict) and asset.get("filename") and not asset.get("url"):
                asset["url"] = asset_url(run_id, str(asset.get("filename") or ""))
        return item

    def payload_for(run_id: str, principal: dict) -> dict:
        with connect() as conn:
            run = require_run(conn, run_id, principal)
            frames = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_frames WHERE run_id=? ORDER BY sequence_index, created_at",
                    (run_id,),
                )
            ]
            pages = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_page_episodes WHERE run_id=? ORDER BY created_at",
                    (run_id,),
                )
            ]
            evidence = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_evidence WHERE run_id=? ORDER BY page_episode_id, created_at",
                    (run_id,),
                )
            ]
            cards = [
                card_row(dict(row), run_id)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_cards WHERE run_id=? ORDER BY created_at",
                    (run_id,),
                )
            ]
        run["metrics"] = _json_object(run.get("metrics"))
        run["client_summary"] = _json_object(run.get("client_summary"))
        for frame in frames:
            frame["accepted"] = bool(frame.get("accepted"))
            frame["frame_meta"] = _json_object(frame.get("frame_meta"))
            if frame.get("filename"):
                frame["url"] = asset_url(run_id, frame["filename"])
        for page in pages:
            page["frame_ids"] = _json_array(page.get("frame_ids"))
            page["coverage_cells"] = _json_array(page.get("coverage_cells"))
            page["payload"] = _json_object(page.get("payload"))
        for item in evidence:
            item["canonical_rect"] = _json_object(item.get("canonical_rect"))
            item["quality"] = _json_object(item.get("quality"))
            item["source_frames"] = _json_array(item.get("source_frames"))
            item["merge_reasons"] = _json_array(item.get("merge_reasons"))
            item["payload"] = _json_object(item.get("payload"))
            if item.get("best_crop_filename"):
                item["best_crop_url"] = asset_url(run_id, item["best_crop_filename"])
        return {"run": run, "frames": frames, "page_episodes": pages, "evidence": evidence, "cards": cards}

    def upsert_manifest(conn, run_id: str, manifest: dict, now: str) -> dict:
        frames = _manifest_list(manifest, "frames", "frame_records")
        pages = _manifest_list(manifest, "page_episodes", "pages", "episodes")
        evidence_items = _manifest_list(manifest, "question_evidence", "evidence", "questionEvidence")
        cards = _manifest_list(manifest, "question_cards", "cards", "questionCards")
        metrics = _json_object(manifest.get("metrics"))
        client_summary = {
            key: manifest.get(key)
            for key in ("source", "input", "out_dir", "simulator_version", "client_version")
            if manifest.get(key) not in (None, "", [], {})
        }
        if metrics:
            conn.execute(
                "UPDATE homework_ledger_runs SET metrics=?, client_summary=?, updated_at=? WHERE id=?",
                (json_dumps(metrics), json_dumps(client_summary), now, run_id),
            )
        for index, item in enumerate(frames, start=1):
            frame_id = _frame_id(item, index)
            filename = _truncate(
                _meta_text(item, "filename", "normalized_filename", "normalizedFilename", "normalized_path", "normalizedPath"),
                260,
            ).replace("\\", "/")
            if filename and "/" in filename:
                filename = _safe_asset_rel(filename, "frames", f"{frame_id}.jpg")
            conn.execute(
                """
                INSERT INTO homework_ledger_frames(
                    id, run_id, sequence_index, filename, original_name, width, height,
                    source_bytes, normalized_bytes, sharpness, brightness, contrast, phash,
                    accepted, reason, page_episode_id, frame_meta, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    sequence_index=excluded.sequence_index, filename=excluded.filename,
                    original_name=excluded.original_name, width=excluded.width, height=excluded.height,
                    source_bytes=excluded.source_bytes, normalized_bytes=excluded.normalized_bytes,
                    sharpness=excluded.sharpness, brightness=excluded.brightness, contrast=excluded.contrast,
                    phash=excluded.phash, accepted=excluded.accepted, reason=excluded.reason,
                    page_episode_id=excluded.page_episode_id, frame_meta=excluded.frame_meta
                """,
                (
                    frame_id,
                    run_id,
                    _int_value(item.get("sequence_index") or item.get("sequenceIndex") or item.get("index")) or index,
                    filename,
                    _truncate(_meta_text(item, "original_name", "originalName", "source_path", "sourcePath"), 260),
                    _int_value(item.get("width")) or 0,
                    _int_value(item.get("height")) or 0,
                    _int_value(item.get("source_bytes") or item.get("sourceBytes")) or 0,
                    _int_value(item.get("normalized_bytes") or item.get("normalizedBytes")) or 0,
                    float(item.get("sharpness") or 0),
                    float(item.get("brightness") or 0),
                    float(item.get("contrast") or 0),
                    _truncate(_meta_text(item, "phash", "visual_hash", "visualHash"), 80),
                    1 if item.get("accepted", True) else 0,
                    _truncate(_meta_text(item, "reason"), 160),
                    _truncate(_meta_text(item, "page_episode_id", "pageEpisodeId"), 80),
                    json_dumps(item),
                    now,
                ),
            )
        for index, item in enumerate(pages, start=1):
            page_id = _truncate(_meta_text(item, "id", "page_episode_id", "pageEpisodeId") or f"page_{index:04d}", 80)
            conn.execute(
                """
                INSERT INTO homework_ledger_page_episodes(
                    id, run_id, first_frame_id, last_frame_id, frame_ids, fingerprint,
                    coverage_cells, payload, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    first_frame_id=excluded.first_frame_id, last_frame_id=excluded.last_frame_id,
                    frame_ids=excluded.frame_ids, fingerprint=excluded.fingerprint,
                    coverage_cells=excluded.coverage_cells, payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    page_id,
                    run_id,
                    _truncate(_meta_text(item, "first_frame_id", "firstFrameId"), 80),
                    _truncate(_meta_text(item, "last_frame_id", "lastFrameId"), 80),
                    json_dumps(_json_array(item.get("frame_ids") or item.get("frameIds"))),
                    _truncate(_meta_text(item, "fingerprint"), 160),
                    json_dumps(_json_array(item.get("coverage_cells") or item.get("coverageCells"))),
                    json_dumps(item),
                    now,
                    now,
                ),
            )
        for index, item in enumerate(evidence_items, start=1):
            evidence_id = _evidence_id(item, index)
            crop_filename = _truncate(
                _meta_text(item, "best_crop_filename", "bestCropFilename", "crop_filename", "cropFilename"),
                260,
            ).replace("\\", "/")
            if crop_filename and "/" in crop_filename:
                crop_filename = _safe_asset_rel(crop_filename, "crops", f"{evidence_id}.jpg")
            conn.execute(
                """
                INSERT INTO homework_ledger_evidence(
                    id, run_id, page_episode_id, best_frame_id, best_crop_filename, crop_kind,
                    canonical_rect, crop_hash, layout_key, ocr_key, seen_count, status,
                    quality, source_frames, merge_reasons, payload, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    page_episode_id=excluded.page_episode_id, best_frame_id=excluded.best_frame_id,
                    best_crop_filename=excluded.best_crop_filename, crop_kind=excluded.crop_kind,
                    canonical_rect=excluded.canonical_rect, crop_hash=excluded.crop_hash,
                    layout_key=excluded.layout_key, ocr_key=excluded.ocr_key, seen_count=excluded.seen_count,
                    status=excluded.status, quality=excluded.quality, source_frames=excluded.source_frames,
                    merge_reasons=excluded.merge_reasons, payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    evidence_id,
                    run_id,
                    _truncate(_meta_text(item, "page_episode_id", "pageEpisodeId"), 80),
                    _truncate(_meta_text(item, "best_frame_id", "bestFrameId"), 80),
                    crop_filename,
                    _truncate(_meta_text(item, "crop_kind", "cropKind") or "question", 40),
                    json_dumps(_rect_payload(item)),
                    _truncate(_meta_text(item, "crop_hash", "cropHash"), 120),
                    _truncate(_meta_text(item, "layout_key", "layoutKey"), 160),
                    _truncate(_meta_text(item, "ocr_key", "ocrKey"), 160),
                    _int_value(item.get("seen_count") or item.get("seenCount")) or 1,
                    _truncate(_meta_text(item, "status") or "ready", 40),
                    json_dumps(_json_object(item.get("quality"))),
                    json_dumps(_json_array(item.get("source_frames") or item.get("sourceFrames"))),
                    json_dumps(_json_array(item.get("merge_reasons") or item.get("mergeReasons"))),
                    json_dumps(item),
                    now,
                    now,
                ),
            )
        for index, item in enumerate(cards, start=1):
            card_id = _card_id(item, index)
            evidence_ids = _json_array(item.get("evidence_ids") or item.get("evidenceIds"))
            figure_assets = _json_array(item.get("figure_assets") or item.get("figureAssets"))
            normalized_assets: list[dict] = []
            for asset in figure_assets:
                if not isinstance(asset, dict):
                    continue
                normalized = dict(asset)
                filename = str(normalized.get("filename") or "").replace("\\", "/").strip().lstrip("/")
                if filename and "/" in filename:
                    normalized["filename"] = _safe_asset_rel(filename, "crops", Path(filename).name)
                normalized_assets.append(normalized)
            conn.execute(
                """
                INSERT INTO homework_ledger_cards(
                    id, run_id, card_key, evidence_ids, number, subject, question_type,
                    stem_text, editable_html, figure_assets, confidence, review_flags,
                    payload, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    card_key=excluded.card_key, evidence_ids=excluded.evidence_ids, number=excluded.number,
                    subject=excluded.subject, question_type=excluded.question_type, stem_text=excluded.stem_text,
                    editable_html=excluded.editable_html, figure_assets=excluded.figure_assets,
                    confidence=excluded.confidence, review_flags=excluded.review_flags,
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    card_id,
                    run_id,
                    _truncate(_meta_text(item, "card_key", "cardKey") or card_id, 160),
                    json_dumps(evidence_ids),
                    _truncate(_meta_text(item, "number"), 80),
                    _truncate(_meta_text(item, "subject"), 80),
                    _truncate(_meta_text(item, "question_type", "questionType", "qtype"), 80),
                    _truncate(_meta_text(item, "stem_text", "stemText", "stem"), 4000),
                    _truncate(_meta_text(item, "editable_html", "editableHtml"), HOMEWORK_LEDGER_HTML_LIMIT),
                    json_dumps(normalized_assets),
                    float(item.get("confidence") or 0),
                    json_dumps(_json_array(item.get("review_flags") or item.get("reviewFlags"))),
                    json_dumps(item),
                    now,
                    now,
                ),
            )
        return {"frames": len(frames), "page_episodes": len(pages), "evidence": len(evidence_items), "cards": len(cards)}

    def render_html(payload: dict) -> str:
        run = payload["run"]
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        cards = payload.get("cards") or []
        evidence_by_id = {item.get("id"): item for item in payload.get("evidence") or []}
        metric_bits = [
            f"frames {html.escape(str(metrics.get('frames_total', len(payload.get('frames') or []))))}",
            f"keyframes {html.escape(str(metrics.get('keyframes', '')))}",
            f"evidence {html.escape(str(metrics.get('question_evidence_count', len(payload.get('evidence') or []))))}",
            f"cards {html.escape(str(metrics.get('question_card_count', len(cards))))}",
            f"upload ratio {html.escape(str(metrics.get('estimated_upload_ratio', '')))}",
        ]
        card_html: list[str] = []
        for index, card in enumerate(cards, start=1):
            title = card.get("number") or f"#{index}"
            flags = " ".join(f"<span>{html.escape(str(flag))}</span>" for flag in card.get("review_flags", [])[:8])
            editable = card.get("editable_html") or f"<article><p>{html.escape(card.get('stem_text') or 'Needs recognition')}</p></article>"
            assets: list[str] = []
            for asset in card.get("figure_assets") or []:
                if not isinstance(asset, dict):
                    continue
                src = asset.get("url") or asset_url(run["id"], str(asset.get("filename") or ""))
                if src:
                    assets.append(
                        f'<figure><img src="{html.escape(src)}" alt="{html.escape(str(asset.get("source_evidence_id") or card.get("id") or ""))}">'
                        f'<figcaption>{html.escape(str(asset.get("type") or "raster"))}</figcaption></figure>'
                    )
            if not assets:
                for evidence_id in card.get("evidence_ids") or []:
                    evidence = evidence_by_id.get(evidence_id)
                    if evidence and evidence.get("best_crop_url"):
                        assets.append(
                            f'<figure><img src="{html.escape(evidence["best_crop_url"])}" alt="{html.escape(str(evidence_id))}">'
                            f'<figcaption>evidence {html.escape(str(evidence_id))}</figcaption></figure>'
                        )
            card_html.append(
                '<section class="card">'
                f'<header><div><strong>{html.escape(str(title))}</strong><small>{html.escape(str(card.get("subject") or "unknown"))} / {html.escape(str(card.get("question_type") or "evidence"))}</small></div>'
                f'<meter min="0" max="1" value="{max(0.0, min(1.0, float(card.get("confidence") or 0))):.3f}"></meter></header>'
                f'<div class="editable" contenteditable="true">{editable}</div>'
                f'<div class="assets">{"".join(assets)}</div>'
                f'<div class="flags">{flags}</div>'
                '</section>'
            )
        return (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(run.get("title") or "Homework Evidence Ledger")}</title>'
            '<style>'
            'body{margin:0;background:#f7f7f4;color:#1d1d1b;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}'
            'main{max-width:1040px;margin:0 auto;padding:18px}'
            'h1{font-size:22px;margin:0 0 8px}.summary{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 18px}.summary span{border:1px solid #d2d2ca;background:#fff;padding:5px 8px;border-radius:6px;font-size:12px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.card{background:#fff;border:1px solid #ddd9ce;border-radius:8px;padding:14px;min-width:0}'
            '.card header{display:flex;justify-content:space-between;gap:12px;align-items:start;border-bottom:1px solid #eee9dd;padding-bottom:8px;margin-bottom:10px}.card small{display:block;color:#6d6a61;margin-top:3px}'
            '.editable{outline:0;line-height:1.55;font-size:15px}.editable:focus{box-shadow:0 0 0 2px #2670ff33;border-radius:4px}.assets{display:grid;gap:8px;margin-top:10px}.assets img{max-width:100%;border:1px solid #dedbd2;border-radius:6px;background:#fafafa}.assets figure{margin:0}.assets figcaption{font-size:11px;color:#777;text-align:center;margin-top:3px}'
            '.flags{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.flags span{background:#edf3ff;color:#22426f;border-radius:6px;padding:3px 6px;font-size:11px}meter{width:80px}'
            '</style></head><body><main>'
            f'<h1>{html.escape(run.get("title") or "Homework Evidence Ledger")}</h1>'
            f'<div class="summary">{"".join(f"<span>{bit}</span>" for bit in metric_bits if bit.strip())}</div>'
            f'<div class="grid">{"".join(card_html) or "<p>No question evidence yet.</p>"}</div>'
            '</main></body></html>'
        )

    @app.post("/api/homework-ledger/runs")
    async def create_homework_ledger_run(request: Request) -> dict:
        init_db()
        principal = principal_from_request(request)
        body = await request.json()
        run_id = _truncate(str(body.get("run_id") or body.get("id") or uuid.uuid4().hex), 80)
        title = clean_user_text(body.get("title") or "Homework Evidence Ledger", 160)
        device_id = clean_user_text(body.get("device_id") or body.get("deviceId") or "iphone", 120)
        student_profile_id = resolve_student_profile(principal["account_id"], str(body.get("student_profile_id") or body.get("studentProfileId") or ""))
        now = utc_now()
        with connect() as conn:
            existing = conn.execute("SELECT * FROM homework_ledger_runs WHERE id=?", (run_id,)).fetchone()
            if existing:
                require_run(conn, run_id, principal)
                conn.execute(
                    """
                    UPDATE homework_ledger_runs
                    SET title=?, device_id=?, student_profile_id=?, updated_at=?
                    WHERE id=?
                    """,
                    (title, device_id, student_profile_id, now, run_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO homework_ledger_runs(
                        id, account_id, created_by_user_id, student_profile_id, device_id,
                        title, status, metrics, client_summary, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, 'running', '{}', '{}', ?, ?)
                    """,
                    (run_id, principal["account_id"], principal.get("user_id", ""), student_profile_id, device_id, title, now, now),
                )
            run = require_run(conn, run_id, principal)
        run_dir(run_id)
        emit_log(f"created homework ledger: {title}", device_id=device_id, source="homework-ledger")
        return {"run_id": run_id, "run": dict(run)}

    @app.get("/api/homework-ledger/runs")
    def list_homework_ledger_runs(request: Request, limit: int = 50) -> dict:
        init_db()
        principal = principal_from_request(request)
        safe_limit = max(1, min(100, int(limit or 50)))
        with connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        runs.*,
                        (SELECT COUNT(*) FROM homework_ledger_frames WHERE run_id=runs.id) AS frame_count,
                        (SELECT COUNT(*) FROM homework_ledger_evidence WHERE run_id=runs.id) AS evidence_count,
                        (SELECT COUNT(*) FROM homework_ledger_cards WHERE run_id=runs.id) AS card_count
                    FROM homework_ledger_runs runs
                    WHERE account_id=?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (principal["account_id"], safe_limit),
                )
            ]
        for row in rows:
            row["metrics"] = _json_object(row.get("metrics"))
            row["client_summary"] = _json_object(row.get("client_summary"))
        return {"runs": rows}

    @app.post("/api/homework-ledger/runs/{run_id}/sync")
    async def sync_homework_ledger_run(
        run_id: str,
        request: Request,
        frames: list[UploadFile] | None = File(None),
        crops: list[UploadFile] | None = File(None),
        manifest: str = Form("{}"),
    ) -> dict:
        init_db()
        principal = principal_from_request(request)
        payload = _parse_manifest(manifest)
        now = utc_now()
        with connect() as conn:
            require_run(conn, run_id, principal)
        frame_uploads = _file_map(frames)
        crop_uploads = _file_map(crops)
        saved_frames = 0
        saved_crops = 0
        for index, item in enumerate(_manifest_list(payload, "frames", "frame_records"), start=1):
            upload_ref = _upload_ref(
                item,
                "upload_ref",
                "uploadRef",
                "filename",
                "normalized_filename",
                "normalizedFilename",
                "normalized_path",
                "normalizedPath",
                fallback=str(index - 1),
            )
            upload = frame_uploads.get(upload_ref) or frame_uploads.get(Path(upload_ref).name)
            if upload is None:
                continue
            frame_id = _frame_id(item, index)
            rel_hint = _upload_ref(
                item,
                "filename",
                "normalized_filename",
                "normalizedFilename",
                "normalized_path",
                "normalizedPath",
                fallback=f"frames/{frame_id}.jpg",
            )
            saved_rel, saved_bytes = await save_upload(upload, run_id=run_id, rel_path=rel_hint, default_folder="frames", fallback_prefix=frame_id)
            item["filename"] = saved_rel
            item["normalized_bytes"] = item.get("normalized_bytes") or item.get("normalizedBytes") or saved_bytes
            saved_frames += 1
        for index, item in enumerate(_manifest_list(payload, "question_evidence", "evidence", "questionEvidence"), start=1):
            upload_ref = _upload_ref(
                item,
                "upload_ref",
                "uploadRef",
                "best_crop_filename",
                "bestCropFilename",
                "crop_filename",
                "cropFilename",
                fallback=str(index - 1),
            )
            upload = crop_uploads.get(upload_ref) or crop_uploads.get(Path(upload_ref).name)
            if upload is None:
                continue
            evidence_id = _evidence_id(item, index)
            rel_hint = _upload_ref(
                item,
                "best_crop_filename",
                "bestCropFilename",
                "crop_filename",
                "cropFilename",
                fallback=f"crops/{evidence_id}.jpg",
            )
            saved_rel, _ = await save_upload(upload, run_id=run_id, rel_path=rel_hint, default_folder="crops", fallback_prefix=evidence_id)
            item["best_crop_filename"] = saved_rel
            saved_crops += 1
        for card in _manifest_list(payload, "question_cards", "cards", "questionCards"):
            for asset in _json_array(card.get("figure_assets") or card.get("figureAssets")):
                if isinstance(asset, dict) and asset.get("filename"):
                    asset["filename"] = _safe_asset_rel(str(asset.get("filename")), "crops", Path(str(asset.get("filename"))).name)
        with connect() as conn:
            require_run(conn, run_id, principal)
            counts = upsert_manifest(conn, run_id, payload, now)
            run = require_run(conn, run_id, principal)
        return {"run_id": run_id, "saved_frames": saved_frames, "saved_crops": saved_crops, "manifest_counts": counts, "run": dict(run)}

    @app.post("/api/homework-ledger/runs/{run_id}/finish")
    async def finish_homework_ledger_run(run_id: str, request: Request) -> dict:
        init_db()
        principal = principal_from_request(request)
        body = await request.json()
        now = utc_now()
        metrics = _json_object(body.get("metrics"))
        with connect() as conn:
            require_run(conn, run_id, principal)
            if metrics:
                conn.execute(
                    "UPDATE homework_ledger_runs SET status='completed', metrics=?, finished_at=?, updated_at=? WHERE id=?",
                    (json_dumps(metrics), now, now, run_id),
                )
            else:
                conn.execute(
                    "UPDATE homework_ledger_runs SET status='completed', finished_at=?, updated_at=? WHERE id=?",
                    (now, now, run_id),
                )
            run = require_run(conn, run_id, principal)
        emit_log(f"finished homework ledger: {run_id}", device_id=run.get("device_id") or "", source="homework-ledger")
        return {"run_id": run_id, "status": run["status"], "run": dict(run)}

    @app.get("/api/homework-ledger/runs/{run_id}")
    def get_homework_ledger_run(run_id: str, request: Request) -> dict:
        init_db()
        principal = principal_from_request(request)
        return payload_for(run_id, principal)

    @app.get("/api/homework-ledger/runs/{run_id}/html", response_class=HTMLResponse)
    def get_homework_ledger_html(run_id: str, request: Request) -> str:
        init_db()
        principal = principal_from_request(request)
        return render_html(payload_for(run_id, principal))

    @app.get("/api/homework-ledger/runs/{run_id}/assets/{asset_path:path}")
    def get_homework_ledger_asset(run_id: str, asset_path: str, request: Request) -> FileResponse:
        init_db()
        principal = principal_from_request(request)
        with connect() as conn:
            require_run(conn, run_id, principal)
        path = asset_file_path(run_id, asset_path)
        media_type = "image/jpeg"
        if path.suffix.lower() == ".png":
            media_type = "image/png"
        elif path.suffix.lower() == ".webp":
            media_type = "image/webp"
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=31536000, immutable"})
