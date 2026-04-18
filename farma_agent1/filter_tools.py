import os
import json
import glob
from langchain_core.tools import tool

# Путь к папке с JSON-инструкциями (можно изменить)
JSON_DIR = 'drug_instructions'
_DRUGS_CACHE = None

def load_all_drugs(directory: str = None) -> list:
    """Загружает все JSON из папки и кэширует."""
    global _DRUGS_CACHE
    if _DRUGS_CACHE is not None:
        return _DRUGS_CACHE
    if directory is None:
        directory = JSON_DIR
    json_files = glob.glob(os.path.join(directory, "*.json"))
    drugs = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['_source_file'] = file
            drugs.append(data)
    _DRUGS_CACHE = drugs
    print(f"[INFO] Загружено {len(drugs)} инструкций из {directory}")
    return drugs

def find_drug_by_priority(drugs, query_name):
    """
    Ищет препарат по приоритету:
    1. Точное совпадение с inn
    2. Точное совпадение с drug_name
    3. Частичное вхождение в inn или drug_name
    4. Поиск по всем разделам (содержимому)
    """
    query_lower = query_name.strip().lower()
    # 1. inn
    for drug in drugs:
        inn = drug.get('inn', '').lower()
        if inn == query_lower:
            return drug
    # 2. drug_name
    for drug in drugs:
        drug_name = drug.get('drug_name', '').lower()
        if drug_name == query_lower:
            return drug
    # 3. Частичное вхождение
    for drug in drugs:
        inn = drug.get('inn', '').lower()
        drug_name = drug.get('drug_name', '').lower()
        if query_lower in inn or query_lower in drug_name:
            return drug
    # 4. По разделам
    for drug in drugs:
        sections = drug.get('sections', {})
        for content in sections.values():
            if content and query_lower in content.lower():
                return drug
    return None

def format_drug_result(drug, sections_list=None, max_section_len=1500):
    """Форматирует вывод препарата, обрезая длинные разделы."""
    result = f"**Препарат:** {drug.get('drug_name')} (МНН: {drug.get('inn')})\n\n"
    sections = drug.get('sections', {})
    if not sections_list:
        for sec, content in sections.items():
            if content:
                title = sec.replace('_', ' ').capitalize()
                if len(content) > max_section_len:
                    content = content[:max_section_len] + "..."
                result += f"--- {title} ---\n{content}\n\n"
    else:
        for sec in sections_list:
            content = sections.get(sec, '')
            title = sec.replace('_', ' ').capitalize()
            if content:
                if len(content) > max_section_len:
                    content = content[:max_section_len] + "..."
                result += f"--- {title} ---\n{content}\n\n"
            else:
                result += f"--- {title} --- (раздел отсутствует)\n\n"
    return result

@tool
def search_drug_by_filters(drug_names=None, sections_list=None) -> str:
    """
    Ищет один или несколько препаратов. drug_names может быть строкой или списком строк.
    """
    print(f"[FILTER] Получены drug_names={drug_names}, sections_list={sections_list}")
    if not drug_names:
        return "Не указано название препарата."
    if isinstance(drug_names, str):
        drug_names = [drug_names]
    drugs = load_all_drugs()
    results = []
    for name in drug_names:
        print(f"[FILTER] Ищем препарат '{name}'...")
        found = find_drug_by_priority(drugs, name)
        if found:
            print(f"[FILTER] Найден: {found.get('drug_name')} (inn: {found.get('inn')})")
            results.append(format_drug_result(found, sections_list))
        else:
            print(f"[FILTER] Препарат '{name}' не найден")
            results.append(f"Препарат '{name}' не найден.")
    return "\n\n=== Следующий препарат ===\n\n".join(results)
