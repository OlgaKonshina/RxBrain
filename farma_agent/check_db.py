"""
Проверка, что база Chroma существует и не пуста.

"""
from chroma_client import get_chroma_collection

collection, _ = get_chroma_collection()
print(f"Коллекция: {collection.name}")
print(f"Количество документов: {collection.count()}")
if collection.count() > 0:
    # Покажем один пример
    sample = collection.peek(limit=1)
    print("\nПример документа:")
    print(sample["documents"][0][:300])
else:
    print("\n База пуста! Нужно проиндексировать данные.")
    print("   Используйте ваш Colab-ноутбук для создания коллекции и сохраните архив.")