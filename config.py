import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
MODEL_CACHE_DIR = "/root/chatbot/models"
# KNOWLEDGE_FILE_PATH = "products.csv"
LLM_MODEL_NAME = "Viet-Mistral/Vistral-7B-Chat"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"