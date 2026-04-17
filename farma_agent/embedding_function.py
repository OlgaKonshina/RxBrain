from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class E5EmbeddingFunction:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = EMBEDDING_MODEL
        print(f"Загрузка модели {model_name}...")
        # для проверки параметры инициализации совпадают со скриптом индексации
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def name(self) -> str:
        return self.model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {text}" for text in input]
        embeddings = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        prefixed = [f"query: {q}" for q in queries]
        embeddings = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
