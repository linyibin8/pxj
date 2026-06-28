import base64
import json
import mimetypes
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from .config import Settings
from . import prompts

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120
MAX_IMAGE_SIDE = 1800
JPEG_QUALITY = 88
TEXT_CONTEXT_TOKEN_LIMIT = 40000
TEXT_TOKEN_SAFETY_MARGIN = 2048
TEXT_MIN_PROMPT_CHARS = 2000
TEXT_FALLBACK_PROMPT_CHARS = 16000
TEXT_FIT_MAX_ATTEMPTS = 6


def _optimized_image_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.getchannel("A")
            background.paste(image.convert("RGB"), mask=alpha)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        out = BytesIO()
        image.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return out.getvalue()


def _image_part(path: Path) -> dict:
    mime = "image/jpeg"
    try:
        image_bytes = _optimized_image_bytes(path)
    except Exception:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def _headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.llm_api_key}"}


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)


def _client(settings: Settings) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(retries=2, trust_env=False)
    return httpx.AsyncClient(
        base_url=settings.llm_base_url,
        timeout=_timeout(),
        transport=transport,
        trust_env=False,
    )


def _truncate_text(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    omitted = len(text) - max_chars
    marker = f"\n...[已按 token 预算截断约 {omitted} 字]...\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return text[:max_chars]
    head_chars = max(1, int(remaining * 0.72))
    tail_chars = max(0, remaining - head_chars)
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    return f"{text[:head_chars].rstrip()}{marker}{tail}"


def _text_payload(settings: Settings, prompt: str, max_tokens: int) -> dict:
    return {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": prompts.get_prompt("text_system")},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _text_prompt_from_payload(payload: dict) -> str:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _set_text_prompt(payload: dict, prompt: str) -> None:
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            message["content"] = prompt
            return


async def _count_payload_tokens(client: httpx.AsyncClient, settings: Settings, payload: dict) -> int | None:
    tokenize_payload = {
        "model": settings.llm_model,
        "messages": payload.get("messages", []),
    }
    if payload.get("chat_template_kwargs"):
        tokenize_payload["chat_template_kwargs"] = payload["chat_template_kwargs"]
    try:
        response = await client.post("/tokenize", json=tokenize_payload, headers=_headers(settings))
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (404, 405):
            return None
        raise
    except httpx.RequestError:
        return None
    data = response.json()
    count = data.get("count")
    if isinstance(count, int):
        return count
    tokens = data.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return None


async def _fit_text_payload_to_context(client: httpx.AsyncClient, settings: Settings, payload: dict, max_tokens: int) -> dict:
    token_limit = max(1024, TEXT_CONTEXT_TOKEN_LIMIT - max_tokens - TEXT_TOKEN_SAFETY_MARGIN)
    count = await _count_payload_tokens(client, settings, payload)
    if count is None:
        prompt = _text_prompt_from_payload(payload)
        if len(prompt) > TEXT_FALLBACK_PROMPT_CHARS:
            _set_text_prompt(payload, f"{prompts.get_prompt('text_token_budget_notice')}\n\n{_truncate_text(prompt, TEXT_FALLBACK_PROMPT_CHARS)}")
        return payload

    for _ in range(TEXT_FIT_MAX_ATTEMPTS):
        if count <= token_limit:
            return payload
        prompt = _text_prompt_from_payload(payload)
        if len(prompt) <= TEXT_MIN_PROMPT_CHARS:
            return payload
        ratio = token_limit / max(count, 1)
        next_chars = max(TEXT_MIN_PROMPT_CHARS, int(len(prompt) * ratio * 0.82))
        next_chars = min(next_chars, len(prompt) - 1)
        shortened = _truncate_text(prompt, next_chars)
        token_budget_notice = prompts.get_prompt("text_token_budget_notice")
        if not shortened.startswith(token_budget_notice):
            shortened = f"{token_budget_notice}\n\n{shortened}"
        _set_text_prompt(payload, shortened)
        count = await _count_payload_tokens(client, settings, payload)
        if count is None:
            return payload
    return payload


def format_llm_error(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return f"无法连接模型服务 {exc.request.url}，请检查模型服务、Tailscale/防火墙或容器网络。原始错误：{exc}"
    if isinstance(exc, httpx.ConnectTimeout):
        return f"连接模型服务超时（{CONNECT_TIMEOUT}s）：{exc.request.url}"
    if isinstance(exc, httpx.ReadTimeout):
        return f"模型响应超时（{READ_TIMEOUT}s）：{exc.request.url}"
    if isinstance(exc, httpx.RemoteProtocolError):
        return f"模型服务连接中断或未返回完整响应：{exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:500].replace("\n", " ")
        return f"模型服务返回 HTTP {exc.response.status_code}：{body}"
    if isinstance(exc, httpx.RequestError):
        return f"请求模型服务失败：{exc}"
    return str(exc)


async def check_health(settings: Settings) -> dict:
    async with _client(settings) as client:
        response = await client.get("/models", headers=_headers(settings))
        response.raise_for_status()
        data = response.json()
    models = [item.get("id", "") for item in data.get("data", [])]
    return {
        "ok": True,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "models": models,
        "model_available": settings.llm_model in models,
    }


async def analyze_images(settings: Settings, prompt: str, image_paths: list[Path]) -> str:
    content: list[dict] = [{"type": "text", "text": f"{prompts.get_prompt('vision_grounding')}\n\n{prompt}"}]
    for index, path in enumerate(image_paths, start=1):
        content.append(
            {
                "type": "text",
                "text": prompts.render_prompt("vision_image_label", index=index, filename=path.name),
            }
        )
        content.append(_image_part(path))
    payload = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": prompts.get_prompt("vision_system"),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 1800,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with _client(settings) as client:
        response = await client.post("/chat/completions", json=payload, headers=_headers(settings))
        response.raise_for_status()
        data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or json.dumps(data, ensure_ascii=False)


async def analyze_images_responses(
    settings: Settings,
    instructions: str,
    prompt: str,
    image_paths: list[Path],
    effort: str = "low",
) -> str:
    """高质量视觉任务（精批/精准分题）走前沿大模型 GPT-5.5 的 Responses API 网关。
    与本地 27B 的 /chat/completions 不同：用 instructions(系统) + input(input_text/input_image) + reasoning.effort。
    返回聚合的 output_text。外部 API，调用方负责并发上限与超时。"""
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for path in image_paths:
        encoded = base64.b64encode(path.read_bytes()).decode()
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"})
    payload = {
        "model": settings.grading_llm_model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": effort},
    }
    timeout = httpx.Timeout(settings.grading_llm_timeout_seconds, connect=CONNECT_TIMEOUT)
    async with httpx.AsyncClient(base_url=settings.grading_llm_url, timeout=timeout, trust_env=False) as client:
        response = await client.post(
            "/v1/responses",
            json=payload,
            headers={"Authorization": f"Bearer {settings.grading_llm_key}", "Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
    out = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    out += part.get("text", "")
    return out or json.dumps(data, ensure_ascii=False)[:2000]


async def analyze_text(settings: Settings, prompt: str, max_tokens: int = 2600) -> str:
    payload = _text_payload(settings, prompt, max_tokens)
    async with _client(settings) as client:
        payload = await _fit_text_payload_to_context(client, settings, payload, max_tokens)
        response = await client.post("/chat/completions", json=payload, headers=_headers(settings))
        response.raise_for_status()
        data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or json.dumps(data, ensure_ascii=False)
