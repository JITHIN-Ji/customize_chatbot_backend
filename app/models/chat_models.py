# backend/app/models/chat_models.py

from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from app.core.config import settings
# In app/models/chat_models.py
from pydantic import BaseModel, HttpUrl
# =================================================================
#  Models for Chat, Documents, and RAG
# =================================================================

class Message(BaseModel):
    role: str # "user", "assistant", or "system"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None
    chatbot_id: int
    llm_provider: Optional[str] = settings.DEFAULT_LLM_PROVIDER
    language: Optional[str] = "auto"
    

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str


class URLIngestRequest(BaseModel):
    """Request model for ingesting content from a URL."""
    url: HttpUrl

# =================================================================
#  Models for User Authentication
# =================================================================

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[EmailStr] = None


class ChatbotCreate(BaseModel):
    chatbotTitle: str
    welcomeMessage: str
    chatbotInstructions: str
    userIconUrl: Optional[str] = None 
    avatarUrl: Optional[str] = None
    bubbleIconUrl: Optional[str] = None
    
    documentId: Optional[str] = None # The ID of the trained document

# This defines the data structure for a single saved chatbot
class Chatbot(BaseModel):
    id: int # The unique ID from the database
    chatbot_title: str
    welcome_message: str
    system_prompt: str
    user_icon_url: Optional[str] = None
    avatar_url: Optional[str] = None
    bubbleIconUrl: Optional[str] = None
    owner_id: int
    document_id: Optional[str] = None
    
    
class ChatbotPublic(BaseModel):
    chatbot_title: str
    welcome_message: str
    avatar_url: Optional[str] = None
    bubble_icon_url: Optional[str] = None
    user_icon_url: Optional[str] = None # <-- ADD THIS LINE
    
    class Config:
        from_attributes = True