import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_CACHE_DIR = "/root/chatbot/models"
LLM_MODEL_NAME = "gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = "data"
DATABASE_URL = os.environ.get("DATABASE_URL")