import os
import json
from langchain_core.tools import tool
from gigachat import GigaChat as GigaChatClient


# Инициализация клиента для работы с файлами
giga_client = GigaChatClient(
    credentials=os.environ["GIGACHAT_CREDENTIALS"],
    verify_ssl_certs=False
)

@tool
def analyze_document(file_path: str) -> dict:
    """
    Анализирует содержимое файла (PDF или изображение) с помощью GigaChat и возвращает JSON.
    """
    if not os.path.exists(file_path):
        return {"error": f"Файл не найден: {file_path}"}

    print(f"[INFO] Загружаем файл в GigaChat: {file_path}")
    try:
        with open(file_path, "rb") as f:
            uploaded_file = giga_client.upload_file(f, purpose="general")
        file_id = uploaded_file.id_
        print(f"[INFO] Файл загружен, ID: {file_id}")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": """
                    Ты — ассистент, который анализирует медицинские документы.
                    Извлеки из файла следующую информацию и верни строго в формате JSON:
                    {
                      "diagnosis": "основной диагноз",
                      "age": "возраст",
                      "pregnancy": "да/нет",
                      "medications": ["лекарство 1", "лекарство 2"],
                      "lab_results": [{"name": "показатель", "value": "значение", "unit": "ед.изм."}],
                      "allergy": "название вещества"
                    }
                    Не добавляй пояснений, только JSON.
                    """,
                    "attachments": [file_id]
                }
            ],
            "model": "GigaChat-2",
            "temperature": 0.1
        }

        response = giga_client.chat(payload)
        content = response.choices[0].message.content
        print(f"[DEBUG] Raw answer: {content}")

        json_start = content.find('{')
        if json_start != -1:
            json_str = content[json_start:]
            return json.loads(json_str)
        else:
            return {"error": "JSON не найден", "raw_content": content}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Ошибка: {str(e)}"}