import os

# RAG
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/home/gleb/iqdoc/hackathon_iqdoc/.chroma_ui")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "iqdoc_ui_v3_minilm")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# GigaChat
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "False")

LLM_MODEL = os.getenv("LLM_MODEL", "GigaChat-2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_SCOPE = os.getenv("LLM_SCOPE", "GIGACHAT_API_PERS")


def _clamp_int(name: str, raw: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, v))


RAG_N_RESULTS = _clamp_int("RAG_N_RESULTS", os.getenv("RAG_N_RESULTS", "25"), 25, 5, 80)
RAG_TOP_K = _clamp_int("RAG_TOP_K", os.getenv("RAG_TOP_K", "8"), 8, 3, 16)
CONTEXT_CHUNK_CHARS = _clamp_int(
    "CONTEXT_CHUNK_CHARS", os.getenv("CONTEXT_CHUNK_CHARS", "1600"), 1600, 400, 3200
)
