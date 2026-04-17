#!/usr/bin/env python3
"""
UI for farma_agent deployment with optional PDF scan upload.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import requests
import streamlit as st


def _resolve_agent_dir() -> Path:
    candidates = [
        Path("/home/gleb/iqdoc/RxBrain-main/farma_agent"),
        Path(__file__).resolve().parents[1] / "RxBrain-main" / "farma_agent",
        Path(__file__).resolve().parents[1] / "farma_agent",
        Path(__file__).resolve().parent / "farma_agent",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Не найден каталог farma_agent.")


AGENT_DIR = _resolve_agent_dir()
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from run_agent import run_agent  # noqa: E402
from rag_tools import search_medical_db  # noqa: E402


def _save_uploaded_pdf(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _cleanup_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _extract_pdf_text_local(file_path: str, max_chars: int = 6000) -> tuple[str, str]:
    """
    Local PDF text extraction for offline mode.
    Tries pdfplumber first, then pypdf as fallback.
    """
    # 1) pdfplumber (better structure for text PDFs)
    try:
        import pdfplumber  # type: ignore

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(f"--- Page {idx} ---\n{text}")
        full_text = "\n\n".join(pages).strip()
        if full_text:
            return full_text[:max_chars], "pdfplumber"
    except Exception:
        pass

    # 2) pypdf fallback
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(file_path)
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"--- Page {idx} ---\n{text}")
        full_text = "\n\n".join(pages).strip()
        if full_text:
            return full_text[:max_chars], "pypdf"
    except Exception:
        pass

    return "", "none"


def _render_result(result: dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return

    st.subheader("Ответ агента")
    st.write(result.get("answer", "Ответ пуст."))

    st.subheader("Источники")
    sources = result.get("sources", [])
    if sources:
        for src in sources:
            st.markdown(f"- {src}")
    else:
        st.info("Источники не возвращены агентом.")

    st.metric("Уверенность", result.get("confidence", "unknown"))

    with st.expander("Полный JSON ответа"):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def _offline_context_limits(model_name: str) -> tuple[int, int, int]:
    """(max_context_chars, max_chunks, max_chars_per_chunk) для локальной LLM."""
    m = model_name.lower()
    if "14b" in m or "13b" in m:
        return 12000, 8, 3600
    if "7b" in m or "8b" in m:
        return 10000, 8, 3200
    if "3b" in m or "4b" in m:
        return 6500, 7, 2400
    return 4200, 5, 1600


def _compress_context(raw_context: str, max_chars: int, max_sources: int) -> str:
    blocks = [b.strip() for b in raw_context.split("\n---\n") if b.strip()]
    selected = blocks[:max_sources]
    compact = "\n---\n".join(selected)
    if len(compact) > max_chars:
        compact = compact[:max_chars] + "\n...\n[контекст сокращён для ускорения генерации]"
    return compact


def _ollama_limits(model_name: str) -> tuple[int, int]:
    """(num_predict, timeout_sec) — крупнее модель, тем дольше CPU-инференс."""
    m = model_name.lower()
    if "14b" in m or "13b" in m:
        return 768, 900
    if "7b" in m or "8b" in m:
        return 768, 600
    if "3b" in m or "4b" in m:
        return 280, 380
    return 160, 220


def _ollama_options(model_name: str, num_predict: int) -> dict:
    """Параметры, которые обычно улучшают связность Qwen на CPU."""
    m = model_name.lower()
    base: dict = {
        "temperature": 0.15,
        "top_p": 0.92,
        "repeat_penalty": 1.08,
        "num_predict": num_predict,
    }
    if "7b" in m or "8b" in m or "14b" in m or "13b" in m:
        base["num_ctx"] = 8192
    elif "3b" in m or "4b" in m:
        base["num_ctx"] = 6144
    else:
        base["num_ctx"] = 4096
    return base


def _ollama_generate(
    model_name: str, prompt: str, num_predict: int, timeout: int
) -> tuple[str, str]:
    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": _ollama_options(model_name, num_predict),
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip(), data.get("done_reason", "")


def _run_offline_agent(
    query: str,
    model_name: str = "qwen2.5:7b",
    top_k_hint: int = 6,
    pdf_text: str = "",
) -> dict:
    # The tool has fixed n_results inside farma_agent, top_k_hint is kept for prompt hinting.
    payload = search_medical_db.invoke({"query": query})
    if not isinstance(payload, dict):
        payload = {}
    chunks = payload.get("chunks", [])
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    max_ctx, max_n_chunk, per_chunk = _offline_context_limits(model_name)
    top_k_hint = min(top_k_hint, max_n_chunk)

    if not chunks:
        context = "По вашему запросу ничего не найдено в базе знаний."
    else:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            meta = c.get("metadata", {})
            blocks.append(
                f"[CHUNK {i}] {c.get('source','source')}\n"
                f"drug={meta.get('drug_name') or meta.get('inn','unknown')}\n"
                f"section={meta.get('section_ru', meta.get('section_type', 'раздел'))}\n"
                f"score={c.get('retrieval_score', 0.0):.3f}\n"
                f"text:\n{c.get('text','')[:per_chunk]}"
            )
        context = _compress_context("\n---\n".join(blocks), max_ctx, max_n_chunk)

    prompt = f"""Ты — клинический фармаколог. Пишешь для врача на русском.
Режим: ТОЛЬКО факты из блоков [CHUNK …] ниже. Если чанков нет или в них нет ответа — напиши в «## Краткий ответ», что по базе данных недостаточно, и не выдумывай дозы и взаимодействия.

Формат ответа (строго Markdown внутри JSON-поля answer):
1) Первая строка поля answer — заголовок: ## Краткий ответ
2) 2–4 предложения: кратко перефразируй суть вопроса, затем главный вывод ТОЛЬКО из чанков.
3) Затем разделы ## с понятными названиями (например «## Дозирование», «## Противопоказания», «## Взаимодействия», «## Мониторинг») — только те, для которых есть текст в CHUNK. В каждом разделе 2–5 маркеров «- » с тезисами; важные дозы/частоты выделяй **жирным**.
4) Не ссылайся на «внешние руководства» и не придумывай исследования — только инструкции из контекста.
5) Поле sources — массив строк: повтори значения строки source из использованных CHUNK (можно 3–8 строк).
6) Поле confidence: high | medium | low — на твой взгляд по полноте чанков относительно вопроса.

Пример ВАЛИДНОГО ответа (короткий образец формата, не копируй содержание):
{{"answer": "## Краткий ответ\\nКратко…\\n\\n## Дозирование\\n- **…** …\\n", "sources": ["препарат / раздел"], "confidence": "medium"}}

Вопрос врача:
{query}

Контекст из базы (фрагменты [CHUNK]):
{context}

Выписка (текст PDF, если есть; иначе игнорируй блок):
{pdf_text if pdf_text else "Не предоставлены."}

Верни ОДИН JSON-объект без текста до/после него. Ключи: answer (строка с Markdown), sources (массив строк), confidence (строка).
"""

    n_pred, t_main = _ollama_limits(model_name)
    n_cont, t_cont = _ollama_limits(model_name)
    n_cont = min(256, max(96, n_cont // 2))

    try:
        raw, done_reason = _ollama_generate(
            model_name=model_name, prompt=prompt, num_predict=n_pred, timeout=t_main
        )
        # If answer is cut by token limit, continue once to complete sentence.
        if done_reason == "length":
            continuation_prompt = (
                "Продолжи ответ с того же места. Заверши мысль и верни только JSON.\n\n"
                f"Текущий ответ:\n{raw}"
            )
            extra, _ = _ollama_generate(
                model_name=model_name, prompt=continuation_prompt, num_predict=n_cont, timeout=t_cont
            )
            raw = f"{raw} {extra}".strip()
    except Exception as exc:
        # Fallback: меньшая модель, если 7b/3b не уложились в таймаут.
        fallbacks = ["qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b"]
        raw = ""
        last_err: Exception | None = exc
        for fb in fallbacks:
            if fb == model_name:
                continue
            try:
                n2, t2 = _ollama_limits(fb)
                raw, _ = _ollama_generate(
                    model_name=fb, prompt=prompt, num_predict=min(n2, 160), timeout=t2
                )
                last_err = None
                break
            except Exception as e2:
                last_err = e2
                continue
        if last_err is not None:
            return {"error": f"Ollama timeout/error: {last_err}"}

    return _parse_offline_response(raw, chunks, metrics if isinstance(metrics, dict) else {})


def _strip_md_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines.pop()
    return "\n".join(lines).strip()


def _sources_from_chunks(chunks: list) -> list[str]:
    out: list[str] = []
    for c in chunks:
        s = c.get("source")
        if s and isinstance(s, str) and s not in out:
            out.append(s)
    return out


def _parse_offline_response(
    raw: str, chunks: list, metrics: dict | None
) -> dict:
    """JSON из ответа Ollama; при сбое — Markdown с чанковыми источниками."""
    cleaned = _strip_md_fence(raw)
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, dict) and obj.get("answer"):
                src = obj.get("sources") or _sources_from_chunks(chunks)
                if not isinstance(src, list):
                    src = [str(src)]
                conf = obj.get("confidence")
                if conf not in ("high", "medium", "low"):
                    conf = (metrics or {}).get("confidence", "medium")
                return {"answer": str(obj["answer"]), "sources": src, "confidence": conf}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    if "##" in cleaned or len(cleaned) > 80:
        return {
            "answer": cleaned,
            "sources": _sources_from_chunks(chunks)
            or ["Локальная база Chroma (retrieval via farma_agent)."],
            "confidence": (metrics or {}).get("confidence", "medium"),
        }

    return {
        "answer": cleaned or "Локальная модель не вернула осмысленный ответ.",
        "sources": _sources_from_chunks(chunks)
        or ["Локальная база Chroma (retrieval via farma_agent)."],
        "confidence": (metrics or {}).get("confidence", "low"),
    }


st.set_page_config(page_title="RxBrain Farma Agent", page_icon="💊", layout="wide")
st.title("💊 RxBrain Farma Agent")
st.caption(
    "Online: LangGraph + GigaChat + Chroma. Offline: Ollama + увеличенный контекст чанков и промпт под Qwen. PDF — локально или через GigaChat."
)

with st.sidebar:
    st.header("Настройки")
    mode = st.radio(
        "Режим работы",
        options=["offline", "online"],
        index=0,
        help="offline — локальный Ollama, online — GigaChat через farma_agent",
    )
    offline_model = st.selectbox(
        "Ollama model (offline)",
        options=["qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b", "qwen2.5:0.5b"],
        index=0,
        help="7b — основной режим на усиленном хосте; меньшие модели — быстрее, проще ответ.",
    )
    gigachat_key = st.text_input(
        "GigaChat API key",
        value=os.getenv("GIGACHAT_CREDENTIALS", ""),
        type="password",
        help="Ключ не сохраняется в коде, только в процессе текущего UI.",
    )
    if gigachat_key.strip():
        os.environ["GIGACHAT_CREDENTIALS"] = gigachat_key.strip()

    st.info("offline: Ollama (по умолчанию qwen2.5:7b); online: GigaChat + PDF-анализ.")

query = st.text_area(
    "Вопрос врача",
    height=130,
    placeholder="Например: Совместимость варфарина и амиодарона у пациента 67 лет с ФП",
)
uploaded_pdf = st.file_uploader(
    "Скан/выписка (PDF, опционально)", type=["pdf"], accept_multiple_files=False
)

run = st.button("Запустить фарма-агент", type="primary", use_container_width=True)

if run:
    if not query.strip():
        st.warning("Введите вопрос врача.")
        st.stop()

    temp_pdf_path = None
    offline_pdf_text = ""
    offline_pdf_parser = ""
    try:
        if uploaded_pdf is not None:
            temp_pdf_path = _save_uploaded_pdf(uploaded_pdf)

        with st.spinner("Агент обрабатывает запрос..."):
            if mode == "offline":
                if temp_pdf_path:
                    offline_pdf_text, offline_pdf_parser = _extract_pdf_text_local(temp_pdf_path)
                    if offline_pdf_text:
                        st.info(
                            f"Локальный PDF-анализ выполнен ({offline_pdf_parser}), "
                            f"извлечено символов: {len(offline_pdf_text)}"
                        )
                    else:
                        st.warning(
                            "Не удалось извлечь текст локально (возможно это скан без OCR). "
                            "Для сканов используйте online-режим или добавьте OCR."
                        )
                result = _run_offline_agent(
                    query=query.strip(),
                    model_name=offline_model,
                    pdf_text=offline_pdf_text,
                )
            else:
                result = run_agent(
                    user_query=query.strip(),
                    file_path=temp_pdf_path,
                    thread_id="streamlit-session",
                )

        _render_result(result)
    finally:
        _cleanup_temp_file(temp_pdf_path)

st.divider()
st.caption(
    "online-режим с PDF требует валидный ключ GigaChat; "
    "offline-режим работает локально через Ollama и локальный PDF parser."
)
