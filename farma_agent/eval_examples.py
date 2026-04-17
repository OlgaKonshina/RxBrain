#!/usr/bin/env python3
"""
Прогон вопросов из IQDOC-файла examples и сравнение с эталоном.

Метрики (реалистичные для RAG):
  - cos_ref: косинусная близость эмбеддингов (answer vs первые ref_prefix символов эталона)
  - term_recall: доля значимых токенов из запроса, встречающихся в ответе (нижний регистр)

Запуск на хосте (нужен доступ к Chroma + GigaChat):
  export GIGACHAT_CREDENTIALS='...'
  export EXAMPLES_PATH=/home/gleb/iqdoc/examples
  cd ~/iqdoc/RxBrain-main/farma_agent && ../venv/bin/python eval_examples.py

Офлайн через Ollama (без GigaChat):
  ../venv/bin/python eval_examples.py --ollama http://127.0.0.1:11434 qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

# farma_agent as cwd
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class ExampleQA:
    idx: int
    qid: str
    specialty: str
    query: str
    reference: str


def parse_examples(path: Path) -> list[ExampleQA]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out: list[ExampleQA] = []
    i = 0
    header_re = re.compile(r"^(\d+)\.\s+(Q\d+)\s*\|\s*(.*)$")
    while i < len(lines):
        m = header_re.match(lines[i].strip())
        if not m:
            i += 1
            continue
        idx = int(m.group(1))
        qid = m.group(2)
        spec = m.group(3).strip()
        i += 1
        query = ""
        ref = ""
        if i < len(lines) and lines[i].startswith("Запрос:"):
            query = lines[i][len("Запрос:") :].strip()
            i += 1
        if i < len(lines) and lines[i].startswith("Ответ:"):
            ref = lines[i][len("Ответ:") :].strip()
            i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(
            ("Источники:", "Предупреждения:")
        ):
            if lines[i].startswith("Запрос:") or lines[i].startswith("Ответ:"):
                break
            i += 1
        if i < len(lines) and lines[i].startswith("Источники:"):
            i += 1
        if i < len(lines) and lines[i].startswith("Предупреждения:"):
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if query:
            out.append(ExampleQA(idx=idx, qid=qid, specialty=spec, query=query, reference=ref))
    return out


def _query_terms(q: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-zА-Яа-яёЁ0-9\-]{4,}", q.lower()) if len(t) >= 4}


def term_recall(answer: str, query: str, max_terms: int = 16) -> float:
    """Доля значимых токенов запроса (до max_terms самых длинных) в ответе."""
    terms = sorted(_query_terms(query), key=len, reverse=True)[:max_terms]
    if not terms:
        return 1.0
    a = answer.lower()
    hit = sum(1 for t in terms if t in a)
    return hit / len(terms)


def cosine_emb(model, a: str, b: str) -> float:
    ea = model.encode(a or "", normalize_embeddings=True)
    eb = model.encode(b or "", normalize_embeddings=True)
    return float(np.dot(ea, eb))


def run_ollama(base_url: str, model: str, prompt: str, num_predict: int = 512, timeout: int = 420) -> str:
    m = model.lower()
    opts: dict = {
        "temperature": 0.15,
        "top_p": 0.92,
        "repeat_penalty": 1.08,
        "num_predict": num_predict,
    }
    if "7b" in m or "8b" in m or "14b" in m:
        opts["num_ctx"] = 8192
    elif "3b" in m or "4b" in m:
        opts["num_ctx"] = 6144
    else:
        opts["num_ctx"] = 4096
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": opts},
        timeout=timeout,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def build_offline_prompt(query: str, chunks_text: str) -> str:
    return f"""Ты — клинический фармаколог. Только факты из [CHUNK] ниже; не выдумывай дозы и исследования.
Формат: JSON с полями answer (Markdown), sources (массив строк source из чанков), confidence.

В answer: начни с ## Краткий ответ; затем 2–4 раздела ## с маркерами «- »; дозы **жирным**.

Вопрос:
{query}

Контекст:
{chunks_text}

Верни один JSON-объект без текста вокруг.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--examples",
        default=os.getenv(
            "EXAMPLES_PATH",
            str(_ROOT.parent.parent / "examples"),
        ),
        help="Путь к файлу examples (IQDOC)",
    )
    ap.add_argument("--limit", type=int, default=20, help="Максимум вопросов")
    ap.add_argument("--ref-prefix", type=int, default=2500, help="Символов эталона для сравнения")
    ap.add_argument("--ollama", nargs=2, metavar=("URL", "MODEL"), help="Оценка через Ollama вместо GigaChat")
    ap.add_argument("--out", default="", help="JSONL с результатами")
    args = ap.parse_args()

    path = Path(args.examples)
    if not path.is_file():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 2

    os.environ.setdefault("DISABLE_RERANK", "1")

    from config import EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer

    from rag_tools import search_medical_db

    examples = parse_examples(path)[: args.limit]
    if not examples:
        print("Не удалось распарсить примеры.", file=sys.stderr)
        return 2

    print(f"Загружено примеров: {len(examples)} из {path}")
    emb_model = SentenceTransformer(EMBEDDING_MODEL)

    rows = []
    cos_vals = []
    rec_vals = []

    for ex in examples:
        print(f"\n=== {ex.idx} {ex.qid} | {ex.specialty} ===")
        print(f"Запрос: {ex.query[:120]}...")

        payload = search_medical_db.invoke({"query": ex.query})
        chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        blocks = []
        for i, c in enumerate(chunks[:10], start=1):
            meta = c.get("metadata", {})
            blocks.append(
                f"[CHUNK {i}] {c.get('source','')}\n"
                f"{c.get('text','')[:2200]}"
            )
        ctx = "\n\n---\n\n".join(blocks) if blocks else "(пусто — ничего не найдено)"

        if args.ollama:
            url, model = args.ollama
            prompt = build_offline_prompt(ex.query, ctx[:12000])
            try:
                raw = run_ollama(url, model, prompt)
            except Exception as e:
                print(f"Ollama error: {e}")
                raw = ""
            answer = raw
            try:
                ja = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
                answer = ja.get("answer", raw)
            except Exception:
                pass
        else:
            from run_agent import run_agent

            if not os.getenv("GIGACHAT_CREDENTIALS"):
                print("Нет GIGACHAT_CREDENTIALS — используйте --ollama URL MODEL", file=sys.stderr)
                return 3
            try:
                res = run_agent(user_query=ex.query, file_path=None, thread_id=f"eval-{ex.qid}")
            except Exception as e:
                print(f"run_agent error: {e}")
                res = {"answer": "", "sources": [], "confidence": "low"}
            answer = res.get("answer") or ""

        ref_slice = ex.reference[: args.ref_prefix]
        cos = cosine_emb(emb_model, answer[:6000], ref_slice)
        rec = term_recall(answer, ex.query)
        cos_vals.append(cos)
        rec_vals.append(rec)
        print(f"cos_ref={cos:.3f} term_recall={rec:.3f} conf={payload.get('metrics',{}) if isinstance(payload, dict) else {}}")

        row = {
            "qid": ex.qid,
            "query": ex.query,
            "cos_ref": cos,
            "term_recall": rec,
            "answer_preview": (answer[:400] + "…") if len(answer) > 400 else answer,
        }
        rows.append(row)

    mean_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
    mean_rec = float(np.mean(rec_vals)) if rec_vals else 0.0
    print(f"\n--- ИТОГО: mean cos_ref={mean_cos:.3f} mean term_recall={mean_rec:.3f} ---")

    if args.out:
        outp = Path(args.out)
        outp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"Сохранено: {outp}")

    # Порог «примерно похоже» — настраивается под RAG (не полный текст эталона)
    target_cos = float(os.getenv("EVAL_TARGET_COS", "0.45"))
    target_rec = float(os.getenv("EVAL_TARGET_REC", "0.18"))
    # Длинные клинические кейсы: часть терминов не повторяется в кратком ответе — допускаем «сильную» семантику.
    strong_semantic = mean_cos >= float(os.getenv("EVAL_STRONG_COS", "0.58"))
    ok = mean_cos >= target_cos and (mean_rec >= target_rec or strong_semantic)
    print(
        f"Цель (env): cos>={target_cos}, recall>={target_rec} "
        f"(или cos>={os.getenv('EVAL_STRONG_COS', '0.58')}) -> {'OK' if ok else 'НИЖЕ ПОРОГА'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
