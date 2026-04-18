import json
import re
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_gigachat import GigaChat
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_SCOPE
from filter_tools import search_drug_by_filters

# Инициализация GigaChat с таймаутом 120 секунд
llm = GigaChat(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    verify_ssl_certs=False,
    scope=LLM_SCOPE,
    timeout=120
)

class PharmaAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    file_summary: dict
    query_summary: dict
    retrieved_chunks: str

def extract_json_from_llm_response(raw: str) -> dict | None:
    """Надёжно извлекает JSON из ответа LLM, обрабатывая экранирование."""
    if not raw:
        return None
    # Если ответ целиком обёрнут в кавычки
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
        try:
            raw = raw.encode().decode('unicode_escape')
        except:
            pass
    # Ищем блок JSON
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1:
        return None
    candidate = raw[start:end+1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Пытаемся убрать неэкранированные переносы строк внутри строк
        candidate = re.sub(r'(?<!\\)\n', ' ', candidate)
        try:
            return json.loads(candidate)
        except:
            return None

def summarize_query_node(state: PharmaAgentState):
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    print(f"[DEBUG summarize] Запрос пользователя: {user_message}")

    prompt_template = """
Ты — помощник для поиска в базе данных инструкций лекарственных препаратов.
База данных содержит JSON-объекты с полями: drug_name, inn и sections (разделы: indications, contraindications, dosage, interactions, side_effects, pregnancy, special_instructions, pharmacology, pharmacokinetics, overdose, storage).

Из запроса пользователя извлеки:
1. Список названий препаратов (МНН или торговые) – если явно перечислены. Если препарат один, верни строку; если несколько – список строк. Если препарат не указан – null.
2. Какие разделы инструкции нужны (список ключей из sections). Если пользователь хочет всю инструкцию или не указал разделы – пустой список.
3. если вотпрос не имеет конкретного преперата предроложи МНН
Верни ТОЛЬКО JSON с полями "drug_names" и "sections".

Правила сопоставления фраз с разделами:
- "противопоказания", "кому нельзя" → contraindications
- "побочные эффекты", "побочка" → side_effects
- "дозировка", "как принимать" → dosage
- "взаимодействие", "с другими лекарствами" → interactions
- "беременность", "кормление грудью" → pregnancy
- "особые указания" → special_instructions
- "фармакология", "механизм действия" → pharmacology
- "фармакокинетика", "как выводится" → pharmacokinetics
- "передозировка" → overdose
- "хранение" → storage
- "показания", "для чего" → indications

Примеры:
Запрос: "Покажи противопоказания для Кинезиа"
Ответ: {{"drug_names": "Кинезиа", "sections": ["contraindications"]}}

Запрос: "Фармакодинамика и фармакокинетика мелоксикама и парацетамола"
Ответ: {{"drug_names": ["мелоксикам", "парацетамол"], "sections": ["pharmacology", "pharmacokinetics"]}}

Запрос: "Взаимодействие мелоксикама с другими НПВС"
Ответ: {{"drug_names": "мелоксикам", "sections": ["interactions"]}}

Запрос: "Дай всю инструкцию на аспирин"
Ответ: {{"drug_names": "аспирин", "sections": []}}

Запрос: "Что лечит Кинезиа?"
Ответ: {{"drug_names": "Кинезиа", "sections": ["indications"]}}

Запрос: "Препарат от рассеянного склероза" (нет названия)
Ответ: {{"drug_names": null, "sections": []}}

Теперь обработай запрос:

Запрос: {user_message}
"""
    prompt = prompt_template.format(user_message=user_message)

    try:
        print("[DEBUG summarize] Вызов LLM...")
        response = llm.invoke([
            SystemMessage(content="Ты — помощник, извлекающий данные. Отвечай только JSON."),
            HumanMessage(content=prompt)
        ])
        raw = response.content.strip()
        print(f"[DEBUG summarize] Ответ LLM: {raw}")
        data = extract_json_from_llm_response(raw)
        if data is None:
            data = {}
    except Exception as e:
        print(f"[DEBUG summarize] Ошибка: {e}")
        data = {}

    drug_names = data.get("drug_names")
    sections = data.get("sections", [])
    print(f"[DEBUG summarize] Извлечено: drug_names={drug_names}, sections={sections}")

    summary = {
        "drug_names": drug_names,
        "sections": sections,
        "main_question": user_message
    }
    return {"query_summary": summary}

def retrieve_node(state: PharmaAgentState):
    print("[DEBUG retrieve] Начало узла retrieve")
    summary = state["query_summary"]
    drug_names = summary.get("drug_names")
    sections = summary.get("sections", [])
    print(f"[DEBUG retrieve] drug_names={drug_names}, sections={sections}")
    if not drug_names:
        result_text = "В запросе не указано название препарата."
    else:
        result_text = search_drug_by_filters.invoke({"drug_names": drug_names, "sections_list": sections})
        # Ограничиваем длину для передачи в LLM (чтобы не было таймаута)
        if len(result_text) > 5000:
            result_text = result_text[:5000] + "\n...[остаток обрезан из-за большого объёма]"
        print(f"[DEBUG retrieve] Результат (первые 300 символов): {result_text[:300]}")
    return {"retrieved_chunks": result_text}

def generate_answer_node(state: PharmaAgentState):
    print("[DEBUG generate] Начало узла generate")
    summary = state["query_summary"]
    chunks = state.get("retrieved_chunks", "")
    file_summary = state.get("file_summary", {})
    user_query = summary.get("main_question", "")

    context = f"""
Данные из эпикриза (учитывай их при рекомендациях):
- Диагноз: {file_summary.get('diagnosis', 'не указан')}
- Возраст: {file_summary.get('age', 'не указан')}
- Беременность: {file_summary.get('pregnancy', 'не указано')}
- Аллергии: {file_summary.get('allergy', 'не указаны')}
- Текущие лекарства: {', '.join(file_summary.get('medications', [])[:10]) if file_summary.get('medications') else 'не указаны'}

Информация из инструкций:
{chunks}

Вопрос пользователя: {user_query} 

Инструкция:
- Ответь на вопрос, используя данные инструкций и эпикриза.
- Если какой-то препарат не найден, сообщи об этом.
- Если запрашивается взаимодействие нескольких препаратов, проанализируй их совместное применение.
- ответ должен быть точным, емким, подробным
- Верни JSON с полями: "answer", "sources" (список названий препаратов), "confidence" (high/medium/low).
"""
    messages = [
        SystemMessage(content="Ты — ассистент клинического фармаколога. Отвечай строго в формате JSON."),
        HumanMessage(content=context)
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    print(f"[DEBUG generate] Сырой ответ LLM (первые 500 символов): {raw[:500]}")

    answer_json = extract_json_from_llm_response(raw)
    if answer_json is None:
        answer_json = {"answer": raw, "sources": [], "confidence": "low"}

    if not isinstance(answer_json.get("sources"), list):
        answer_json["sources"] = []
    if "confidence" not in answer_json:
        answer_json["confidence"] = "medium"

    return {"messages": [AIMessage(content=json.dumps(answer_json, ensure_ascii=False))]}

# Сборка графа
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