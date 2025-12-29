import numpy as np
from typing import List, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, gemini_client=None): 
        self.embedding_model = None
        if settings.GEMINI_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=settings.GEMINI_EMBEDDING_MODEL_NAME,
                    google_api_key=settings.GEMINI_API_KEY,  
                    task_type="retrieval_document" 
                )
                logger.info(f"EmbeddingGenerator configured to use LangChain Gemini model: {settings.GEMINI_EMBEDDING_MODEL_NAME}")
            except ImportError:
                logger.error("Package 'langchain-google-genai' not found. Please install it to use embeddings.")
                self.embedding_model = None
            except Exception as e:
                logger.error(f"Failed to initialize GoogleGenerativeAIEmbeddings: {e}")
                self.embedding_model = None
        else:
            logger.error("GEMINI_API_KEY not found. EmbeddingGenerator is not functional.")

    

    async def generate_embeddings(self, texts: List[str], task_type: str = "retrieval_document") -> Optional[np.ndarray]:
        if not self.embedding_model:
            logger.error("No embedding model is available.")
            return None

        try:
            
            # This code now assumes we are always using the Gemini model.
            if task_type == "retrieval_query":
                # embed_query expects a single string, not a list
                embeddings = self.embedding_model.embed_query(texts[0]) 
                return np.array([embeddings], dtype=np.float32)
            else:
                embeddings = self.embedding_model.embed_documents(texts)
                return np.array(embeddings, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return None

