import os

# RAG
CHROMA_DB_PATH = "farma_agent/chroma_db2"
COLLECTION_NAME = "iqdoc_baseline"
EMBEDDING_MODEL =  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# GigaChat
os.environ["GIGACHAT_CREDENTIALS"] = "" #  ваш ключ
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"

LLM_MODEL = "GigaChat-2-Pro"
LLM_TEMPERATURE = 0.1
LLM_SCOPE = "GIGACHAT_API_PERS"
