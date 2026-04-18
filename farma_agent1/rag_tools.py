from langchain_core.tools import tool
from chroma_client import get_chroma_collection
from typing import Annotated, TypedDict, Literal


@tool
def search_medical_db(query: str) -> str:
    """
    Ищет информацию в базе данных инструкций к лекарствам и справочнике ГРЛС.
    Возвращает форматированный текст с указанием источника и релевантности.
    """
    collection, embedding_fn = get_chroma_collection()
    if collection.count() == 0:
        return "База знаний пуста. Сначала необходимо проиндексировать документы."

    query_emb = embedding_fn.embed_queries([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results= 15,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        return "По вашему запросу ничего не найдено в базе знаний."

    formatted_chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        score = 1 - dist
        drug = meta.get("drug_name") or meta.get("inn") or meta.get("source_type", "unknown")
        section = meta.get("section_ru", "раздел")
        # Формируем читаемый источник
        source_text = f"Инструкция к препарату «{drug}», раздел «{section}»"

        header = f"=== ИСТОЧНИК: {source_text} (релевантность: {score:.3f}) ==="
        # Ограничим длину текста, чтобы не перегружать контекст
        text_preview = doc[:1200] + "..." if len(doc) > 1200 else doc
        formatted_chunks.append(f"{header}\n{text_preview}\n---\n")

    return "\n".join(formatted_chunks)