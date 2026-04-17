# agent_graph.py
import json
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_gigachat import GigaChat
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_SCOPE
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


# ----------------------------------------------------------------------
# Узел 1: Формируем ТОЛЬКО поисковые термины (без эпикриза)
# ----------------------------------------------------------------------
def summarize_query_node(state: PharmaAgentState):
    # Находим последний запрос пользователя
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    prompt = f"""Ты — ассистент, который формулирует **поисковые запросы** для базы c инструкции к лекарствам.

Запрос врача: {user_message}

Твоя задача: выделить ключевые медицинские понятия для поиска. 
- Если запрос о замене препарата: укажи класс препарата (например, "гепатопротектор") и название препарата, который заменяем (если есть).
- Если запрос о взаимодействии двух препаратов: укажи оба названия.
- перефразируй запрос под семантический поиск по векторной базе
Верни ТОЛЬКО JSON с полями:
- "search_terms": содежит массив не более 6 слов Не используй кавычки. Не добавляй пояснения.
- "medications": список конкретных лекарств, упомянутых в запросе (если есть). Напиши к какой группе относится препарат.

"""

    response = llm.invoke([
        SystemMessage(content="Ты — помощник, извлекающий поисковые термины. Отвечай только JSON."),
        HumanMessage(content=prompt)
    ])
    search_phrase = response.content.strip().strip('"').strip("'")
    # Очистка от лишних пробелов
    search_phrase = ' '.join(search_phrase.split())
    if not search_phrase or len(search_phrase) > 200:
        search_phrase = user_message
    print(f"[DEBUG] Поисковая фраза: {search_phrase}")

    summary = {
        "search_terms": [search_phrase],
        "main_question": user_message,
        "medications": []
    }
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
        chunks = search_medical_db.invoke({"query": query_for_search})
        print(f"[DEBUG] Длина ответа поиска: {len(chunks)} символов")
    except Exception as e:
        chunks = f"Ошибка поиска: {e}"
    return {"retrieved_chunks": chunks}


# ----------------------------------------------------------------------
# Узел 3: Генерация ответа с учётом эпикриза
# ----------------------------------------------------------------------
def generate_answer_node(state: PharmaAgentState):
    summary = state["query_summary"]
    chunks = state.get("retrieved_chunks", "")
    file_summary = state.get("file_summary", {})
    user_query = summary.get("main_question", "")

    if not chunks or "Ничего не найдено" in chunks or "Ошибка" in chunks:
        chunks = "По вашему запросу ничего не найдено в базе медицинских знаний."

    # Формируем контекст, включая эпикриз
    context = f"""
дополнительные данные (учитывай их при выборе препарата):
- Диагноз: {file_summary.get('diagnosis', 'не указан')}
- Возраст: {file_summary.get('age', 'не указан')}
- Беременность: {file_summary.get('pregnancy', 'не указано')}
- Аллергии: {file_summary.get('allergy', 'не указаны')}
- Текущие лекарства: {', '.join(file_summary.get('medications', [])[:10]) if file_summary.get('medications') else 'не указаны'}

РЕЗУЛЬТАТЫ ПОИСКА ПО БАЗЕ ЗНАНИЙ (инструкции):
{chunks}

ИНСТРУКЦИЯ:
- Ответь на вопрос пользователя: {user_query}
- Используй результаты поиска для информации о препаратах (дозировки, противопоказания, взаимодействия).
- Учитывай данные эпикриза (возраст, беременность, аллергии, текущие лекарства) для персонализации ответа. 
- Если в эпикризе есть противопоказания беременность, аллергия, возраст , обязательно учти их.  
- Если в результатах поиска нет нужной информации, скажи об этом. 
- в ответе напиши возраста можно использвать препарат, учитывай другие препараты пациента и диагноз, содержашиеся в документе (если есть)
- Верни JSON с полями: "answer", "sources" (список строк), "confidence" (high/medium/low).
"""
    messages = [
        SystemMessage(content="Ты — ассистент клинического фармаколога. Отвечай строго в формате JSON."),
        HumanMessage(content=context)
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    try:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            json_str = raw[start:end + 1]
            answer_json = json.loads(json_str)
        else:
            answer_json = {"answer": raw, "sources": [], "confidence": "low"}
    except Exception:
        answer_json = {"answer": raw, "sources": [], "confidence": "low"}

    if not isinstance(answer_json.get("sources"), list):
        answer_json["sources"] = []
    if "confidence" not in answer_json:
        answer_json["confidence"] = "medium"

    return {"messages": [AIMessage(content=json.dumps(answer_json, ensure_ascii=False))]}


# ----------------------------------------------------------------------
# Сборка графа
# ----------------------------------------------------------------------
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
