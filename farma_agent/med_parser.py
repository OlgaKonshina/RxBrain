#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Универсальный парсер медицинских документов (PDF).
Поддерживает:
- Текстовые PDF (извлечение через pdfplumber, таблицы в Markdown)
- Отсканированные PDF (OCR через EasyOCR)
- Автоматическое определение типа документа
"""

import os
import io
import numpy as np
import fitz  
import pdfplumber
import easyocr
from PIL import Image
from typing import Tuple, Optional


OCR_LANGUAGES = ['ru', 'en']  
OCR_GPU = False  
OCR_DPI = 150  



def is_text_pdf(pdf_path: str, check_pages: int = 3) -> bool:
  
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(min(check_pages, doc.page_count)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                doc.close()
                return True
        doc.close()
        return False
    except Exception as e:
        print(f"Ошибка при проверке PDF: {e}")
        return False


def parse_with_pdfplumber(pdf_path: str) -> str:
    
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Извлечение обычного текста
            page_text = page.extract_text()
            if page_text:
                full_text.append(f"--- Страница {page_num} ---")
                full_text.append(page_text.strip())

            # Извлечение таблиц
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    table_md = table_to_markdown(table)
                    if table_md:
                        full_text.append("\n[ТАБЛИЦА]\n" + table_md + "\n[/ТАБЛИЦА]")
    return "\n\n".join(full_text)


def table_to_markdown(table: list) -> str:
    """Преобразует список списков (таблицу) в Markdown-строку."""
    if not table or len(table) == 0:
        return ""

    # Определяем максимальное количество столбцов
    num_cols = max(len(row) for row in table if row) if table else 0
    if num_cols == 0:
        return ""

    # Приводим все строки к одинаковой длине, заменяя None на пустую строку
    clean_table = []
    for row in table:
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        if len(clean_row) < num_cols:
            clean_row.extend([""] * (num_cols - len(clean_row)))
        clean_table.append(clean_row)

    # Формируем Markdown
    md_rows = []
    for i, row in enumerate(clean_table):
        md_rows.append("| " + " | ".join(row) + " |")
        if i == 0 and len(clean_table) > 1:
            # Разделитель заголовка
            separator = ["---"] * num_cols
            md_rows.append("| " + " | ".join(separator) + " |")
    return "\n".join(md_rows)


def parse_with_easyocr(pdf_path: str) -> str:
    """
    Извлечение текста из отсканированного PDF с помощью EasyOCR.
    """
    # Инициализируем reader (один раз, можно вынести глобально)
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num in range(doc.page_count):
        print(f"  Обработка страницы {page_num + 1} из {doc.page_count}...")
        page = doc[page_num]
        # Конвертируем страницу в изображение с заданным DPI
        pix = page.get_pixmap(dpi=OCR_DPI)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        # Преобразуем PIL Image в numpy array (EasyOCR требует именно такой формат)
        img_np = np.array(img)

        # Распознаём текст
        result = reader.readtext(img_np, detail=0, paragraph=True)
        page_text = "\n".join(result)
        if page_text.strip():
            all_text.append(f"--- Страница {page_num + 1} ---")
            all_text.append(page_text.strip())

    doc.close()
    return "\n\n".join(all_text)


def pdf_parser(pdf_path: str, force_ocr: bool = False) -> Tuple[str, str]:
  
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")

    # Принудительный OCR
    if force_ocr:
        print(" Принудительное использование EasyOCR...")
        return parse_with_easyocr(pdf_path), "easyocr"

    # Автоматическое определение
    if is_text_pdf(pdf_path):
        print(" Обнаружен текстовый PDF. Используем pdfplumber...")
        return parse_with_pdfplumber(pdf_path), "pdfplumber"
    else:
        print(" Похоже на скан. Используем EasyOCR...")
        return parse_with_easyocr(pdf_path), "easyocr"


if __name__ == "__main__":
   

    pdf_file = 'путь к файлу'
    

    try:
        text, parser_type = pdf_parser(pdf_file)
        print("\n" + "=" * 50)
        print(f"РЕЗУЛЬТАТ (парсер: {parser_type})")
        print("=" * 50)
        # Выводим первые 2000 символов для просмотра
        if len(text) > 2000:
            print(text[:2000])
            print(f"\n... (всего символов: {len(text)})")
        else:
            print(text)

        # сохранить результат в файл
        output_file = os.path.splitext(pdf_file)[0] + "_extracted.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\n Полный текст сохранён в файл: {output_file}")

    except Exception as e:
        print(f" Ошибка: {e}")
        sys.exit(1)
