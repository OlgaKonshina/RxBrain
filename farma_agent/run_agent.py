import json
import os
from agent_graph import app
from file_tools import analyze_document
from langchain_core.messages import HumanMessage, SystemMessage


def run_agent(user_query: str, file_path: str = None, thread_id: str = "1"):
    print("=== run_agent: начало ===")
    print(f"Запрос: {user_query}")
    print(f"Путь к файлу: {file_path}")

    # Анализ файла, если передан
    file_summary = {}
    if file_path and os.path.exists(file_path):
        print(f"[INFO] Анализируем файл через analyze_document: {file_path}")
        result = analyze_document.invoke({"file_path": file_path})
        if "error" in result:
            print(f"[ERROR] Ошибка при анализе файла: {result['error']}")
            return {"error": result['error']}
        file_summary = result
        print(f"[INFO] Результат analyze_document: {json.dumps(file_summary, ensure_ascii=False, indent=2)}")
    elif file_path:
        print(f"[ERROR] Файл не найден: {file_path}")
        return {"error": f"Файл не найден: {file_path}"}
    else:
        print("[INFO] Файл не передан.")

    print(f"[DEBUG] initial_state['file_summary'] содержит: {bool(file_summary)} записей")

    # Начальное состояние графа
    initial_state = {
        "messages": [
            SystemMessage(content="Ты — ассистент клинического фармаколога."),
            HumanMessage(content=user_query),
        ],
        "file_summary": file_summary,
        "query_summary": {},
        "retrieved_chunks": "",
        "retrieval_payload": {},
    }

    print("[INFO] Запуск графа...")
    final_state = app.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    print("[INFO] Граф завершён.")

    # Извлекаем последнее сообщение (ответ генерации)
    last_message = final_state["messages"][-1]
    try:
        result = json.loads(last_message.content)
    except Exception as e:
        print(f"[ERROR] Не удалось распарсить ответ: {e}")
        result = {"answer": last_message.content, "sources": [], "confidence": "low"}

    print(f"[DEBUG] Ответ агента: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result


if __name__ == "__main__":
    user_query = ("Фармакодинамика и фармакокинетика препарата мелоксикам + механика взаимодействия с другими нпвс и парацетамолом")
    file_path = ""  # путь к файлу
    if os.path.exists(file_path):
        result = run_agent(user_query, file_path=file_path)
        print("\n=== ИТОГОВЫЙ ОТВЕТ ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = run_agent(user_query)
