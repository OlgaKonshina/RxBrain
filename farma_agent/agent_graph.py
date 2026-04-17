# agent_graph.py
import json
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_gigachat import GigaChat

from config import CONTEXT_CHUNK_CHARS, LLM_MODEL, LLM_SCOPE, LLM_TEMPERATURE, RAG_TOP_K
from rag_tools import search_medical_db

llm = GigaChat(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    verify_ssl_certs=False,
    scope=LLM_SCOPE,
)


class PharmaAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    file_summary: dict
    query_summary: dict
    retrieved_chunks: str
    retrieval_payload: dict


# ----------------------------------------------------------------------
# Узел 1: Формируем ТОЛЬКО поисковые термины (без эпикриза)
# ----------------------------------------------------------------------
def summarize_query_node(state: PharmaAgentState):
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    prompt = f"""Ты — ассистент, который формулирует **поисковые запросы** для базы с инструкциями к лекарствам.

Запрос врача: {user_message}

Твоя задача: выделить ключевые медицинские понятия для поиска.
- Если запрос о замене препарата: укажи класс препарата (например, "гепатопротектор") и название препарата, который заменяем (если есть).
- Если запрос о взаимодействии двух препаратов: укажи оба названия.
- перефразируй запрос под семантический поиск по векторной базе
Верни ТОЛЬКО JSON с полями:
- "search_terms": массив коротких строк для поиска (не более 6 элементов).
- "medications": список конкретных лекарств, упомянутых в запросе (если есть).

"""

    response = llm.invoke(
        [
            SystemMessage(content="Ты — помощник, извлекающий поисковые термины. Отвечай только JSON."),
            HumanMessage(content=prompt),
        ]
    )
    raw = (response.content or "").strip()
    summary: dict = {
        "search_terms": [],
        "main_question": user_message,
        "medications": [],
    }
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            obj = json.loads(raw[start : end + 1])
            terms = obj.get("search_terms", [])
            if isinstance(terms, str):
                terms = [terms]
            elif not isinstance(terms, list):
                terms = []
            clean_terms: list[str] = []
            for t in terms:
                if isinstance(t, str) and t.strip():
                    clean_terms.append(" ".join(t.split()))
            if clean_terms:
                summary["search_terms"] = clean_terms
            meds = obj.get("medications", [])
            if isinstance(meds, list):
                summary["medications"] = [m for m in meds if isinstance(m, str) and m.strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    if not summary["search_terms"]:
        fallback = " ".join(user_message.split())
        summary["search_terms"] = [fallback[:400] if len(fallback) > 400 else fallback]

    joined = " ".join(summary["search_terms"])
    print(f"[DEBUG] Поисковые термины ({len(summary['search_terms'])}): {joined[:200]}")
    return {"query_summary": summary}


# ----------------------------------------------------------------------
# Узел 2: Поиск в базе знаний
# ----------------------------------------------------------------------
def retrieve_node(state: PharmaAgentState):
    summary = state["query_summary"]
    search_terms = summary.get("search_terms", [])
    if not search_terms:
        search_terms = [summary.get("main_question", "")]
    query_for_search = " ".join(search_terms)
    print(f"[DEBUG] Поисковый запрос в базе: {query_for_search}")

    try:
        payload = search_medical_db.invoke({"query": query_for_search})
        if isinstance(payload, dict) and payload.get("error"):
            chunks = f"Ошибка поиска: {payload['error']}"
            return {"retrieved_chunks": chunks, "retrieval_payload": {}}

        chunk_items = payload.get("chunks", []) if isinstance(payload, dict) else []
        if not chunk_items:
            return {
                "retrieved_chunks": "По вашему запросу ничего не найдено в базе знаний.",
                "retrieval_payload": payload if isinstance(payload, dict) else {},
            }

        blocks = []
        for i, c in enumerate(chunk_items, start=1):
            meta = c.get("metadata", {})
            drug = meta.get("drug_name") or meta.get("inn") or "unknown"
            inn = meta.get("inn", "не указан")
            section = meta.get("section_ru", meta.get("section_type", "раздел"))
            score = c.get("retrieval_score", 0.0)
            rerank_score = c.get("rerank_score", 0.0)
            source = c.get("source", f"{drug} / {section}")
            text = c.get("text", "")
            block = (
                f"[CHUNK {i}] source={source}\n"
                f"drug={drug}\ninn={inn}\nsection={section}\n"
                f"retrieval_score={score:.3f}\nrerank_score={rerank_score:.3f}\n"
                f"text:\n{text[:CONTEXT_CHUNK_CHARS]}"
            )
            blocks.append(block)

        chunks = "\n\n---\n\n".join(blocks)
        print(f"[DEBUG] Retrieval chunks prepared: {len(chunk_items)}")
        return {"retrieved_chunks": chunks, "retrieval_payload": payload}
    except Exception as e:
        chunks = f"Ошибка поиска: {e}"
        return {"retrieved_chunks": chunks, "retrieval_payload": {}}


# ----------------------------------------------------------------------
# Узел 3: Генерация ответа с учётом эпикриза
# ----------------------------------------------------------------------
def generate_answer_node(state: PharmaAgentState):
    summary = state["query_summary"]
    chunks = state.get("retrieved_chunks", "")
    retrieval_payload = state.get("retrieval_payload", {}) or {}
    file_summary = state.get("file_summary", {})
    user_query = summary.get("main_question", "")

    if not chunks or "Ничего не найдено" in chunks or "Ошибка" in chunks:
        chunks = "По вашему запросу ничего не найдено в базе медицинских знаний."

    context = f"""
дополнительные данные из медицинской выписки (обязательно учитывай их при ответе):
- Диагноз: {file_summary.get('diagnosis', 'не указан')}
- Возраст: {file_summary.get('age', 'не указан')}
- Беременность: {file_summary.get('pregnancy', 'не указано')}
- Аллергии: {file_summary.get('allergy', 'не указаны')}
- Текущие лекарства: {', '.join(file_summary.get('medications', [])[:10]) if file_summary.get('medications') else 'не указаны'}

РЕЗУЛЬТАТЫ ПОИСКА ПО БАЗЕ ЗНАНИЙ (инструкции):
{chunks}

ИНСТРУКЦИЯ:
- Ответь на вопрос пользователя: {user_query}
- Используй ТОЛЬКО результаты поиска по базе знаний для информации о препаратах (дозировки, противопоказания, взаимодействия).
- Учитывай данные выписки (возраст, беременность, аллергии, текущие лекарства) для персонализации ответа.
- Если в выписке есть факторы риска (беременность, аллергия, возрастные ограничения, сопутствующая терапия), обязательно отрази это в ответе.
- Если в результатах поиска нет нужной информации, скажи об этом явно.
- Не проси пользователя дополнительно указать международное название (МНН), если вопрос уже задан по торговому названию.
- Не добавляй сведения из внешних источников или общих знаний вне найденных фрагментов базы.
- Стиль как в датасете IQDOC / examples: Markdown с заголовками ## и ###, списки «- », жирное для доз и критичных фраз.
- Обязательно начни поле answer с блока «## Краткий ответ»: одно короткое предложение, перефразирующее клиническую ситуацию из вопроса (используй термины из вопроса, где уместно), затем 2–5 предложений с главным выводом из базы.
- Далее 2–5 разделов ## по сути вопроса (например: «## Механизм действия», «## Дозирование», «## Противопоказания», «## Взаимодействия», «## Мониторинг») — только если это следует из фрагментов базы.
- Где уместно, в конце абзаца укажи в скобках ссылку на препарат и тип раздела из контекста (как в эталонах: «(препарат, раздел …)»).
- Отдельно «## Риски и предупреждения» с учётом выписки пациента, если в выписке есть факторы риска; иначе кратко из контекста базы.
- «## Практические действия» — конкретные шаги, вытекающие только из контекста.
- Не дублируй весь текст инструкции; выбирай релевантное. Без фактов вне найденных CHUNK.
- Верни JSON с полями:
  - "answer": структурированный Markdown по пунктам выше,
  - "sources": список строк-источников (препарат + раздел),
  - "confidence": high/medium/low.
"""
    messages = [
        SystemMessage(content="Ты — ассистент клинического фармаколога. Отвечай строго в формате JSON."),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    draft = response.content.strip()

    fact_check_prompt = f"""
Ты редактор медицинского ответа. Оставь только утверждения, подтверждаемые контекстом базы.
Нельзя добавлять новые факты. Сохрани Markdown-структуру (##, ###, списки, **жирный**).
Удаляй или сокращай только абзацы/предложения без опоры в CHUNK; не ломай заголовок «## Краткий ответ».

КОНТЕКСТ ИЗ БАЗЫ:
{chunks}

ЧЕРНОВИК ОТВЕТА:
{draft}

Верни только JSON:
{{
  "answer": "<очищенный ответ>",
  "removed_claims": ["кратко что убрано"]
}}
"""
    checked_resp = llm.invoke(
        [
            SystemMessage(content="Проверяй факты строго по контексту базы, без домыслов."),
            HumanMessage(content=fact_check_prompt),
        ]
    )
    raw = checked_resp.content.strip() if checked_resp and checked_resp.content else draft
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            json_str = raw[start : end + 1]
            answer_json = json.loads(json_str)
        else:
            answer_json = {"answer": raw, "sources": [], "confidence": "low"}
    except Exception:
        answer_json = {"answer": raw, "sources": [], "confidence": "low"}

    sources = []
    for c in retrieval_payload.get("chunks", [])[:RAG_TOP_K]:
        src = c.get("source")
        if src and src not in sources:
            sources.append(src)
    answer_json["sources"] = sources

    calibrated_conf = retrieval_payload.get("metrics", {}).get("confidence")
    if calibrated_conf in {"high", "medium", "low"}:
        answer_json["confidence"] = calibrated_conf
    elif "confidence" not in answer_json:
        answer_json["confidence"] = "medium"

    return {"messages": [AIMessage(content=json.dumps(answer_json, ensure_ascii=False))]}


workflow = StateGraph(PharmaAgentState)
workflow.add_node("summarize", summarize_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_answer_node)

workflow.set_entry_point("summarize")
workflow.add_edge("summarize", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
