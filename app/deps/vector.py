from typing import Optional
from app.services.vector_store_manager import VectorStoreManager

def get_vs(user_id: str, dim: Optional[int] = None):
    return VectorStoreManager(user_id=user_id, dimension=dim)
