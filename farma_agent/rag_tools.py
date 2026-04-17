import os
import re
from typing import Any

from langchain_core.tools import tool
from sentence_transformers import CrossEncoder

from chroma_client import get_chroma_collection
from config import RAG_N_RESULTS, RAG_TOP_K

_RERANKER = None
_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _get_reranker():
    global _RERANKER
    if os.getenv("DISABLE_RERANK", "").lower() in ("1", "true", "yes"):
        return False
    if _RERANKER is None:
        try:
            _RERANKER = CrossEncoder(_RERANKER_MODEL)
        except Exception as exc:
            print(f"[WARN] Не удалось загрузить reranker {_RERANKER_MODEL}: {exc}")
            _RERANKER = False
    return _RERANKER


def _query_rewrites(query: str) -> list[str]:
    q = " ".join(query.split())
    q_norm = re.sub(r"[^\w\s\-]", " ", q.lower())
    q_norm = " ".join(q_norm.split())
    q_clinical = f"{q} противопоказания взаимодействия дозировка ограничения мониторинг"
    variants = [q, q_norm, q_clinical]
    unique = []
    for v in variants:
        if v and v not in unique:
            unique.append(v)
    return unique


def _extract_query_entities(query: str) -> set[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9\-]{4,}", query.lower())
    return set(tokens)


def _chunk_id(doc: str, meta: dict[str, Any]) -> str:
    drug_key = meta.get("drug_key", "")
    section = meta.get("section_type", "")
    return f"{drug_key}|{section}|{doc[:80]}"


def _normalize_entities(chunks: list[dict[str, Any]], query_entities: set[str]) -> dict[str, str]:
    mapping = {}
    for c in chunks:
        drug_name = (c["metadata"].get("drug_name") or "").lower()
        inn = (c["metadata"].get("inn") or "").lower()
        if not drug_name or not inn:
            continue
        for token in query_entities:
            if token in drug_name:
                mapping[token] = inn
    return mapping


def _calc_metrics(chunks: list[dict[str, Any]], query_entities: set[str], entity_map: dict[str, str]) -> dict[str, Any]:
    if not chunks:
        return {
            "avg_retrieval_score": 0.0,
            "avg_rerank_score": 0.0,
            "entity_coverage": 0.0,
            "contradiction_flags": 0,
            "confidence": "low",
        }

    avg_score = sum(c["retrieval_score"] for c in chunks) / len(chunks)
    rerank_vals = [c.get("rerank_score", c["retrieval_score"]) for c in chunks]
    avg_rerank = sum(rerank_vals) / len(rerank_vals)
    coverage = len(entity_map) / max(1, len(query_entities))

    contradiction_flags = 0
    for c in chunks:
        t = c["text"].lower()
        if "противопоказ" in t and "показан" in t:
            contradiction_flags += 1

    if avg_score >= 0.62 and coverage >= 0.45 and contradiction_flags == 0:
        confidence = "high"
    elif avg_score >= 0.48 and coverage >= 0.25:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "avg_retrieval_score": round(avg_score, 4),
        "avg_rerank_score": round(avg_rerank, 4),
        "entity_coverage": round(coverage, 4),
        "contradiction_flags": contradiction_flags,
        "confidence": confidence,
    }


@tool
def search_medical_db(query: str) -> dict:
    """
    Поиск только по локальной базе: query-rewrite, retrieval (RAG_N_RESULTS), rerank (RAG_TOP_K).
    Возвращает структурированный payload для генерации без внешних источников.
    """
    collection, embedding_fn = get_chroma_collection()
    if collection.count() == 0:
        return {"error": "База знаний пуста. Сначала необходимо проиндексировать документы."}

    rewrites = _query_rewrites(query)
    query_entities = _extract_query_entities(query)
    candidates = {}

    for q in rewrites:
        query_emb = embedding_fn.embed_queries([q])[0]
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=RAG_N_RESULTS,
            include=["documents", "metadatas", "distances"],
        )
        if not results.get("documents") or not results["documents"][0]:
            continue

        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            rid = _chunk_id(doc, meta)
            score = max(0.0, 1.0 - float(dist))
            if rid not in candidates or score > candidates[rid]["retrieval_score"]:
                candidates[rid] = {
                    "text": doc,
                    "metadata": meta,
                    "retrieval_score": score,
                }

    if not candidates:
        return {
            "query": query,
            "rewrites": rewrites,
            "chunks": [],
            "normalized_entities": {},
            "metrics": _calc_metrics([], query_entities, {}),
        }

    candidate_list = list(candidates.values())

    reranker = _get_reranker()
    if reranker:
        pairs = [(query, c["text"][:1400]) for c in candidate_list]
        rr_scores = reranker.predict(pairs)
        for c, rr in zip(candidate_list, rr_scores):
            c["rerank_score"] = float(rr)
        candidate_list.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        for c in candidate_list:
            c["rerank_score"] = c["retrieval_score"]
        candidate_list.sort(key=lambda x: x["retrieval_score"], reverse=True)

    top_chunks = candidate_list[:RAG_TOP_K]
    for c in top_chunks:
        c["source"] = (
            f"{c['metadata'].get('drug_name') or c['metadata'].get('inn') or 'unknown'}"
            f" / {c['metadata'].get('section_ru', c['metadata'].get('section_type', 'раздел'))}"
        )

    entity_map = _normalize_entities(top_chunks, query_entities)
    metrics = _calc_metrics(top_chunks, query_entities, entity_map)

    return {
        "query": query,
        "rewrites": rewrites,
        "chunks": top_chunks,
        "normalized_entities": entity_map,
        "metrics": metrics,
    }
