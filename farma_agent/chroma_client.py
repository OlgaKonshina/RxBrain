import chromadb
from embedding_function import E5EmbeddingFunction
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL


def load_chroma_collection():
    embedding_fn = E5EmbeddingFunction(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
        print(f"✅ Коллекция '{COLLECTION_NAME}' загружена. Чанков: {collection.count()}")
    except Exception as e:
        print(f"⚠️ Коллекция не найдена, создаём новую (будет пустой): {e}")
        collection = client.create_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    return collection, embedding_fn


# Глобальная переменная для коллекции
_chroma_collection = None
_embedding_fn = None


def get_chroma_collection():
    global _chroma_collection, _embedding_fn
    if _chroma_collection is None:
        _chroma_collection, _embedding_fn = load_chroma_collection()
    return _chroma_collection, _embedding_fn
