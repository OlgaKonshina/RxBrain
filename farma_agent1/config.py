import os

# RAG


EMBEDDING_MODEL =  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# GigaChat
os.environ["GIGACHAT_CREDENTIALS"] = "" #  ваш ключ
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"

LLM_MODEL = "GigaChat-2"
LLM_TEMPERATURE = 0.3
LLM_SCOPE = "GIGACHAT_API_PERS"
