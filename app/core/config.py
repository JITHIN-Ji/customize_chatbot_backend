from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    
    PROJECT_NAME: str = "Chatbot API"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key_that_should_be_long_and_random")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 # Token now expires in 60 minutes (1 hour)
    GOOGLE_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")
    CLIENT_ORIGIN_URL: str = os.getenv("CLIENT_ORIGIN_URL", "http://localhost:3000")

    # --- General LLM & Embedding Settings --- 
    # Generic LLM API Key (optional, specific keys below are preferred)
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") 

    # Default embedding model (Sentence Transformers, used if Gemini embedding is not configured)
    # DEFAULT_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Gemini Settings --- 
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"  # Or your preferred Gemini generation model
    GEMINI_EMBEDDING_MODEL_NAME: str = "models/text-embedding-004" # Or your preferred Gemini embedding model

    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

    OPENROUTER_SITE_URL: Optional[str] = os.getenv("OPENROUTER_SITE_URL")
    OPENROUTER_SITE_NAME: Optional[str] = os.getenv("OPENROUTER_SITE_NAME")

    

    # --- Groq Settings --- 
    # GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    # GROQ_MODEL_NAME: str = "llama3-8b-8192" # Or your preferred Groq model (e.g., mixtral-8x7b-32768)

    # --- Storage and Processing Settings --- 
    # Data directory where source documents are stored (can be overridden by .env)
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # Vector store path (can be overridden by .env)
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store_data/faiss_index")

    # Upload directory (can be overridden by .env)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # FAISS metadata file name
    FAISS_METADATA_FILE: str = "faiss_metadata.json"

    # Chunking parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    TOP_K: int = 5 

    # --- Agent Settings ---
    DEFAULT_LLM_PROVIDER: str = "openai" # Can be 'gemini' or 'groq'

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields from .env that are not defined in Settings

settings = Settings()

# Determine active embedding model based on API key availability
ACTIVE_EMBEDDING_MODEL_NAME = settings.GEMINI_EMBEDDING_MODEL_NAME
USING_GEMINI_EMBEDDINGS = bool(settings.GEMINI_API_KEY)
print(f"[Config] UPLOAD_DIR: {settings.UPLOAD_DIR}")
print(f"[Config] VECTOR_STORE_PATH: {settings.VECTOR_STORE_PATH}")
print(f"[Config] Active embedding model: {ACTIVE_EMBEDDING_MODEL_NAME}")
print(f"[Config] Using Gemini embeddings: {USING_GEMINI_EMBEDDINGS}")
print(f"[Config] Gemini API Key Loaded: {'Yes' if settings.GEMINI_API_KEY else 'No'}")
print(f"[Config] OpenAI API Key Loaded: {'Yes' if settings.OPENROUTER_API_KEY else 'No'}")
